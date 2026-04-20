#!/usr/bin/env python3
"""CLI entry point for Decipher — run benchmarks or crack ciphers headlessly."""
from __future__ import annotations

import argparse
import os
import sys


def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import keyring
        key = keyring.get_password("decipher", "anthropic_api_key")
        if key:
            return key
    except Exception:
        pass
    print("Error: No API key found. Set ANTHROPIC_API_KEY or configure via the keychain.", file=sys.stderr)
    sys.exit(1)


def cmd_benchmark(args: argparse.Namespace) -> None:
    from benchmark.loader import BenchmarkLoader

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

    if args.automated_only:
        from automated.runner import AutomatedBenchmarkRunner

        runner = AutomatedBenchmarkRunner(
            verbose=args.verbose,
            language=args.language,
            artifact_dir=args.artifact_dir or "artifacts",
        )
        model = "automated-only"
    else:
        from benchmark.runner_v2 import BenchmarkRunnerV2
        from services.claude_api import ClaudeAPI

        api_key = get_api_key()
        model = args.model or "claude-opus-4-7"
        api = ClaudeAPI(api_key=api_key, model=model)
        runner = BenchmarkRunnerV2(
            claude_api=api,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
            language=args.language,
            artifact_dir=args.artifact_dir or "artifacts",
        )

    print(f"Running {len(tests)} test(s) — model={model}, max_iter={args.max_iterations}")
    print(f"Artifacts → {args.artifact_dir or 'artifacts'}/<test_id>/<run_id>.json\n")

    results = []
    for i, test in enumerate(tests):
        print(f"[{i+1}/{len(tests)}] {test.test_id} — {test.description}")
        test_data = loader.load_test_data(test)
        result = runner.run_test(test_data)
        conf = f"{result.self_confidence:.2f}" if result.self_confidence is not None else "n/a"
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

    if results:
        n = len(results)
        avg_char = sum(r.char_accuracy for r in results) / n
        avg_word = sum(r.word_accuracy for r in results) / n
        solved = sum(1 for r in results if r.status == "solved")
        print(f"AVERAGE: {solved}/{n} declared solutions, char={avg_char:.1%}, word={avg_word:.1%}")


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

    if args.automated_only:
        from automated.runner import run_automated, save_crack_artifact

        print("Running automated-only solver (no LLM API calls)...")
        artifact = run_automated(
            cipher_text=ct,
            language=args.language,
            cipher_id=cipher_id,
        )
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
    from services.claude_api import ClaudeAPI

    api_key = get_api_key()
    model = args.model or "claude-opus-4-7"
    api = ClaudeAPI(api_key=api_key, model=model)

    def on_event(event: str, payload: dict) -> None:
        if event == "iteration_start":
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
        verbose=args.verbose,
        on_event=on_event,
    )

    path = Path(artifact_dir) / cipher_id / f"{artifact.run_id}.json"
    try:
        artifact.save(path)
        print(f"\nArtifact saved: {path}")
    except Exception as e:  # noqa: BLE001
        print(f"\nWarning: failed to save artifact: {e}")

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
    print(f"\nFinal decryption ({final_branch}):\n{final_dec}")


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
    if args.seed is not None:
        spec.seed = args.seed

    if args.flush_cache:
        n = cache.flush(spec)
        print(f"Flushed {n} cache entry for this spec.")

    if args.automated_only:
        cached = cache.get(spec)
        if cached is None:
            print(
                "Error: --automated-only testgen cannot generate new plaintext, "
                "because that would require an LLM API call. Run without "
                "--automated-only once to populate the cache, or choose a cached spec.",
                file=sys.stderr,
            )
            sys.exit(1)
        api_key = ""
    else:
        api_key = get_api_key()

    test_data = build_test_case(spec, cache, api_key, seed=args.seed)

    pt_preview = test_data.plaintext[:120] + ("..." if len(test_data.plaintext) > 120 else "")
    ct_preview = test_data.canonical_transcription[:120] + "..."
    print(f"Test ID:   {test_data.test.test_id}")
    print(f"Plaintext: {pt_preview}")
    print(f"Cipher:    {ct_preview}")
    print(f"Desc:      {test_data.test.description}")

    if args.dry_run:
        print("\n[dry-run] Skipping agent.")
        return

    if args.automated_only:
        from automated.runner import AutomatedBenchmarkRunner

        runner = AutomatedBenchmarkRunner(
            verbose=args.verbose,
            language=args.language,
            artifact_dir=args.artifact_dir,
        )
        print("\nRunning automated-only solver (no LLM API calls)...")
        result = runner.run_test(test_data, language=args.language)
    else:
        from benchmark.runner_v2 import BenchmarkRunnerV2
        from services.claude_api import ClaudeAPI

        crack_model = args.model or "claude-opus-4-7"
        crack_api = ClaudeAPI(api_key=api_key, model=crack_model)
        runner = BenchmarkRunnerV2(
            claude_api=crack_api,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
            language=args.language,
            artifact_dir=args.artifact_dir,
        )
        print(f"\nRunning agent (model={crack_model}, max_iter={args.max_iterations})...")
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

    # benchmark
    bench = subparsers.add_parser("benchmark", help="Run benchmark tests against historical datasets")
    bench.add_argument("benchmark_path", help="Path to benchmark root directory")
    bench.add_argument("--split", "-s", help="Split file name (default: auto-detect from source)")
    bench.add_argument("--track", "-t", default="transcription2plaintext")
    bench.add_argument("--source", help="Source filter (e.g. 'borg', 'copiale')")
    bench.add_argument("--test-id", help="Run a single test by ID")
    bench.add_argument("--limit", "-n", type=int, help="Maximum number of tests to run")
    bench.add_argument("--max-iterations", "-i", type=int, default=25)
    bench.add_argument("--model", "-m", help="Claude model (default: claude-opus-4-7)")
    bench.add_argument("--language", "-l", choices=["en", "la", "de", "fr", "it", "unknown"])
    bench.add_argument("--artifact-dir", help="Artifact output directory (default: ./artifacts)")
    bench.add_argument("--verbose", "-v", action="store_true")
    bench.add_argument("--automated-only", action="store_true",
                       help="Run native automated solvers only; make no LLM API calls.")

    # crack
    crack = subparsers.add_parser("crack", help="Crack a cipher from file or stdin")
    crack.add_argument("--file", "-f", help="Input file (default: stdin)")
    crack.add_argument("--canonical", action="store_true",
                       help="Input is canonical S-token format (space-separated, | word breaks)")
    crack.add_argument("--max-iterations", "-i", type=int, default=25)
    crack.add_argument("--model", "-m", help="Claude model (default: claude-opus-4-7)")
    crack.add_argument("--language", "-l", choices=["en", "la", "de", "fr", "it", "unknown"],
                       default="en")
    crack.add_argument("--artifact-dir", help="Artifact output directory (default: ./artifacts)")
    crack.add_argument("--cipher-id", help="Identifier for this cipher (default: 'cli')")
    crack.add_argument("--verbose", "-v", action="store_true")
    crack.add_argument("--automated-only", action="store_true",
                       help="Run native automated solvers only; make no LLM API calls.")

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
    tg.add_argument("--seed", type=int)
    tg.add_argument("--flush-cache", action="store_true")
    tg.add_argument("--flush-all-cache", action="store_true")
    tg.add_argument("--list-cache", action="store_true")
    tg.add_argument("--dry-run", action="store_true")
    tg.add_argument("--max-iterations", "-i", type=int, default=25)
    tg.add_argument("--model", "-m")
    tg.add_argument("--artifact-dir", default="artifacts")
    tg.add_argument("--cache-dir", default="testgen_cache")
    tg.add_argument("--verbose", "-v", action="store_true")
    tg.add_argument("--automated-only", action="store_true",
                    help=(
                        "Run native automated solvers only; make no LLM API calls. "
                        "Requires the generated plaintext to already be cached."
                    ))

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {"benchmark": cmd_benchmark, "crack": cmd_crack, "testgen": cmd_testgen}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
