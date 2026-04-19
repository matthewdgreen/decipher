#!/usr/bin/env python3
"""CLI entry point for Decipher — run benchmarks or crack ciphers headlessly."""
from __future__ import annotations

import argparse
import json
import os
import sys


def get_api_key() -> str:
    """Get API key from environment or keychain."""
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
    print("Error: No API key found. Set ANTHROPIC_API_KEY env var or configure via the GUI.", file=sys.stderr)
    sys.exit(1)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run benchmark tests."""
    # Lazy imports to avoid loading Qt until needed
    from PySide6.QtCore import QCoreApplication

    # Need a QCoreApplication for QObject-based Session/Settings
    qt_app = QCoreApplication.instance()
    if qt_app is None:
        qt_app = QCoreApplication(sys.argv[:1])

    from benchmark.loader import BenchmarkLoader
    from benchmark.scorer import ScoreResult, format_report, score_decryption
    from services.claude_api import ClaudeAPI

    api_key = get_api_key()
    model = args.model or "claude-opus-4-7"
    api = ClaudeAPI(api_key=api_key, model=model)

    if args.v2:
        _run_benchmark_v2(args, api)
        return

    from benchmark.runner import BenchmarkRunner

    loader = BenchmarkLoader(args.benchmark_path)

    # Determine which split file to use
    split_file = args.split
    if not split_file:
        # Default: try source-specific split file
        if args.source:
            split_file = f"{args.source}_tests.jsonl"
        else:
            split_file = "all_tests.jsonl"

    tests = loader.load_tests(
        split_file,
        track=args.track,
        source=args.source,
    )

    if args.test_id:
        tests = [t for t in tests if t.test_id == args.test_id]

    if not tests:
        print("No matching tests found.", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        tests = tests[:args.limit]

    print(f"Running {len(tests)} benchmark test(s)...")
    print(f"Model: {model}")
    print(f"Max iterations: {args.max_iterations}")
    print()

    runner = BenchmarkRunner(
        claude_api=api,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        language=args.language,
    )

    scores: list[ScoreResult] = []
    for i, test in enumerate(tests):
        print(f"[{i+1}/{len(tests)}] {test.test_id} — {test.description}")

        test_data = loader.load_test_data(test)
        result = runner.run_test(test_data)

        print(f"  Status: {result.status}, "
              f"Agent score: {result.best_score:.1%}, "
              f"Iterations: {result.iterations_used}, "
              f"Time: {result.elapsed_seconds:.1f}s")

        # Score against ground truth
        score = score_decryption(
            test_id=result.test_id,
            decrypted=result.final_decryption,
            ground_truth=test_data.plaintext,
            agent_score=result.best_score,
            status=result.status,
        )
        scores.append(score)

        print(f"  Char accuracy: {score.char_accuracy:.1%}, "
              f"Word accuracy: {score.word_accuracy:.1%}")

        if result.error_message:
            print(f"  Error: {result.error_message}")

        if args.verbose and result.final_decryption:
            preview = result.final_decryption[:200]
            print(f"  Decryption: {preview}")

        print()

    # Print summary report
    print(format_report(scores))

    # Optionally save results to JSON
    if args.output:
        results_data = []
        for s in scores:
            results_data.append({
                "test_id": s.test_id,
                "status": s.status,
                "char_accuracy": round(s.char_accuracy, 4),
                "word_accuracy": round(s.word_accuracy, 4),
                "agent_score": round(s.agent_score, 4),
                "total_chars": s.total_chars,
                "correct_chars": s.correct_chars,
                "total_words": s.total_words,
                "correct_words": s.correct_words,
            })
        with open(args.output, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {args.output}")


def _build_cipher_from_input(text: str, canonical: bool):
    """Parse text input into a CipherText, shared between v1 and v2 crack."""
    from benchmark.runner import parse_canonical_transcription
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText

    if canonical:
        return parse_canonical_transcription(text)
    ignore = {" ", "\t", "\n", "\r"}
    alphabet = Alphabet.from_text(text, ignore_chars=ignore)
    clean = " ".join(text.split())
    return CipherText(raw=clean, alphabet=alphabet, source="cli", separator=" ")


def _run_benchmark_v2(args: argparse.Namespace, api) -> None:
    """Run the v2 workspace-based agent across the selected benchmark tests."""
    from benchmark.loader import BenchmarkLoader
    from benchmark.runner_v2 import BenchmarkRunnerV2

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

    runner = BenchmarkRunnerV2(
        claude_api=api,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        language=args.language,
        artifact_dir=args.artifact_dir or "artifacts",
    )

    print(f"Running {len(tests)} v2 test(s) with model {api.model}")
    print(f"Max iterations: {args.max_iterations}")
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
            f"Self-conf: {conf}, "
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
        print(f"AVERAGE: {solved}/{n} declared solutions, "
              f"char={avg_char:.1%}, word={avg_word:.1%}")


def _run_crack_v2(args: argparse.Namespace, api, ct) -> None:
    """Run the v2 agent on a single CLI-supplied cipher."""
    from agent.loop_v2 import run_v2
    from benchmark.scorer import score_decryption  # used only if ground truth supplied

    cipher_id = args.cipher_id or "cli"
    artifact_dir = args.artifact_dir or "artifacts"

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

    # Save artifact
    from pathlib import Path
    path = Path(artifact_dir) / cipher_id / f"{artifact.run_id}.json"
    try:
        artifact.save(path)
        print(f"\nArtifact saved: {path}")
    except Exception as e:  # noqa: BLE001
        print(f"\nWarning: failed to save artifact: {e}")

    # Show summary
    print(f"Status: {artifact.status}")
    if artifact.solution:
        print(f"Declared branch: {artifact.solution.branch}")
        print(f"Self-confidence: {artifact.solution.self_confidence:.2f}")
        print(f"Rationale: {artifact.solution.rationale}")
    iterations = (
        max(tc.iteration for tc in artifact.tool_calls) if artifact.tool_calls else 0
    )
    print(f"Iterations: {iterations}")
    print(f"Tool calls: {len(artifact.tool_calls)}")

    final_branch = artifact.solution.branch if artifact.solution else "main"
    final_dec = next(
        (b.decryption for b in artifact.branches if b.name == final_branch),
        artifact.branches[0].decryption if artifact.branches else "",
    )
    print(f"\nFinal decryption ({final_branch}):\n{final_dec}")


def cmd_crack(args: argparse.Namespace) -> None:
    """Crack a cipher from a file or stdin."""
    from PySide6.QtCore import QCoreApplication

    qt_app = QCoreApplication.instance()
    if qt_app is None:
        qt_app = QCoreApplication(sys.argv[:1])

    from models.session import Session
    from services.claude_api import ClaudeAPI

    api_key = get_api_key()
    model = args.model or "claude-opus-4-7"
    api = ClaudeAPI(api_key=api_key, model=model)

    # Read input
    if args.file:
        with open(args.file) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("Error: No input text provided.", file=sys.stderr)
        sys.exit(1)

    ct = _build_cipher_from_input(text, args.canonical)

    print(f"Alphabet: {ct.alphabet.size} symbols, {len(ct.tokens)} tokens, {len(ct.words)} words")

    if args.v2:
        _run_crack_v2(args, api, ct)
        return

    # Run the v1 agent
    import time
    from agent.prompts import get_system_prompt, initial_context
    from agent.state import AgentState
    from agent.tools import TOOL_DEFINITIONS, ToolExecutor
    from analysis import dictionary, frequency, ic, pattern

    lang = args.language
    dict_path = dictionary.get_dictionary_path(lang)
    word_set = dictionary.load_word_set(dict_path)
    word_list = pattern.load_word_list(dict_path)
    pattern_dict = pattern.build_pattern_dictionary(word_list)

    session = Session()
    session.set_cipher_text(ct)

    state = AgentState(max_iterations=args.max_iterations)
    executor = ToolExecutor(session, word_set, pattern_dict, agent_state=state)

    freq_data = frequency.sorted_frequency(ct.tokens)
    freq_display = [
        (ct.alphabet.symbol_for(tid), count, count / len(ct.tokens) * 100)
        for tid, count in freq_data
    ]
    ic_value = ic.index_of_coincidence(ct.tokens, ct.alphabet.size)
    pre_mapping_score = dictionary.score_plaintext(ct.raw.upper(), word_set)

    context = initial_context(
        cipher_display=ct.raw,
        alphabet_symbols=ct.alphabet.symbols,
        freq_data=freq_display,
        ic_value=ic_value,
        total_tokens=len(ct.tokens),
        pre_mapping_score=pre_mapping_score,
        language=lang,
    )

    state.status = "running"
    state.add_user_message(context)

    start = time.time()
    for iteration in range(state.max_iterations):
        state.iteration = iteration + 1
        pre_score = dictionary.score_plaintext(session.apply_key(), word_set)
        state.save_checkpoint(pre_score, session.key)
        state.previous_score = pre_score

        if args.verbose:
            print(f"Iteration {state.iteration}, score={pre_score:.2%}")

        response = api.send_message(
            messages=state.messages,
            tools=TOOL_DEFINITIONS,
            system=get_system_prompt(lang),
        )

        assistant_content = []
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                if args.verbose:
                    print(f"  Agent: {block.text[:200]}")
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input,
                })
                tool_uses.append({
                    "id": block.id, "name": block.name, "input": block.input,
                })

        state.add_assistant_message(assistant_content)

        if tool_uses:
            tool_results = []
            for tool in tool_uses:
                result = executor.execute(tool["name"], tool["input"])
                tool_results.append((tool["id"], result))
                if args.verbose:
                    print(f"  Tool: {tool['name']}")

            decrypted = session.apply_key()
            score = dictionary.score_plaintext(decrypted, word_set)
            state.update_best(score, session.key)

            if state.score_dropped(score):
                drop_warning = (
                    f"\n⚠️ WARNING: Score DROPPED from {state.previous_score:.2%} "
                    f"to {score:.2%}. Call rollback to undo."
                )
                last_id, last_result = tool_results[-1]
                tool_results[-1] = (last_id, last_result + drop_warning)

            state.add_tool_results(tool_results)

            if score >= 0.98 and session.is_complete:
                state.status = "solved"
                break
        else:
            state.status = "solved" if state.best_score >= 0.5 else "stuck"
            break

    if state.status == "running":
        state.status = "stuck"

    if state.best_score > dictionary.score_plaintext(session.apply_key(), word_set):
        session.set_full_key(state.best_key)

    elapsed = time.time() - start
    final = session.apply_key()

    print(f"\nStatus: {state.status}")
    print(f"Score: {state.best_score:.2%}")
    print(f"Iterations: {state.iteration}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nDecrypted text:\n{final}")


def cmd_testgen(args: argparse.Namespace) -> None:
    """Generate a synthetic cipher test case and optionally run the v2 agent."""
    from benchmark.runner_v2 import BenchmarkRunnerV2
    from benchmark.scorer import score_decryption
    from services.claude_api import ClaudeAPI
    from testgen.builder import build_test_case
    from testgen.cache import PlaintextCache
    from testgen.spec import DifficultyPreset, TestSpec

    api_key = get_api_key()
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

    # Build spec
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

    # benchmark subcommand
    bench = subparsers.add_parser("benchmark", help="Run benchmark tests")
    bench.add_argument(
        "benchmark_path",
        help="Path to benchmark root directory",
    )
    bench.add_argument(
        "--split", "-s",
        help="Split file name (default: auto-detect from source)",
    )
    bench.add_argument(
        "--track", "-t",
        default="transcription2plaintext",
        help="Track filter (default: transcription2plaintext)",
    )
    bench.add_argument(
        "--source",
        help="Source filter (e.g. 'borg', 'copiale')",
    )
    bench.add_argument(
        "--test-id",
        help="Run a single test by ID",
    )
    bench.add_argument(
        "--limit", "-n",
        type=int,
        help="Maximum number of tests to run",
    )
    bench.add_argument(
        "--max-iterations", "-i",
        type=int,
        default=25,
        help="Max agent iterations per test (default: 25)",
    )
    bench.add_argument(
        "--model", "-m",
        help="Claude model to use (default: claude-opus-4-7)",
    )
    bench.add_argument(
        "--output", "-o",
        help="Save results to JSON file",
    )
    bench.add_argument(
        "--language", "-l",
        choices=["en", "la", "de", "fr", "it", "unknown"],
        help="Target plaintext language (default: auto-detect from benchmark metadata)",
    )
    bench.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output (show agent reasoning)",
    )
    bench.add_argument(
        "--v2",
        action="store_true",
        help="Use the v2 workspace-based agent (branches, signal panel, agent-driven termination).",
    )
    bench.add_argument(
        "--artifact-dir",
        help="Directory to write v2 run artifacts (default: ./artifacts)",
    )

    # crack subcommand
    crack = subparsers.add_parser("crack", help="Crack a cipher from file or stdin")
    crack.add_argument(
        "--file", "-f",
        help="Input file (default: read from stdin)",
    )
    crack.add_argument(
        "--canonical",
        action="store_true",
        help="Input is canonical S-token format (space-separated, | word breaks)",
    )
    crack.add_argument(
        "--max-iterations", "-i",
        type=int,
        default=25,
        help="Max agent iterations (default: 25)",
    )
    crack.add_argument(
        "--model", "-m",
        help="Claude model to use",
    )
    crack.add_argument(
        "--language", "-l",
        choices=["en", "la", "de", "fr", "it", "unknown"],
        default="en",
        help="Target plaintext language (default: en)",
    )
    crack.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    crack.add_argument(
        "--v2",
        action="store_true",
        help="Use the v2 workspace-based agent.",
    )
    crack.add_argument(
        "--artifact-dir",
        help="Directory to write v2 run artifact (default: ./artifacts).",
    )
    crack.add_argument(
        "--cipher-id",
        help="Identifier for this cipher (used in v2 artifact path). Defaults to 'cli'.",
    )

    # gui subcommand (launch the GUI)
    subparsers.add_parser("gui", help="Launch the GUI application")

    # testgen subcommand
    tg = subparsers.add_parser(
        "testgen",
        help="Generate a synthetic test case and optionally run the v2 agent on it",
    )
    tg.add_argument("--language", "-l", choices=["en", "it", "de", "fr", "la"], default="en")
    tg.add_argument("--preset", "-p", choices=["tiny", "easy", "medium", "hard"], default="easy")
    tg.add_argument("--length", type=int, help="Override approx word count from preset")
    tg.add_argument("--topic", default="general", help="Topic hint for LLM generation")
    tg.add_argument("--no-boundaries", action="store_true", help="No word boundaries in cipher")
    tg.add_argument("--seed", type=int, help="Seed for cipher key RNG (reproducible runs)")
    tg.add_argument("--flush-cache", action="store_true", help="Delete cache entry for this spec before running")
    tg.add_argument("--flush-all-cache", action="store_true", help="Delete entire testgen_cache/ before running")
    tg.add_argument("--list-cache", action="store_true", help="Print cached entries and exit")
    tg.add_argument("--dry-run", action="store_true", help="Generate/cache plaintext, print it, skip agent")
    tg.add_argument("--max-iterations", "-i", type=int, default=25)
    tg.add_argument("--model", "-m", help="Claude model for cracking (default: claude-opus-4-7)")
    tg.add_argument("--artifact-dir", default="artifacts")
    tg.add_argument("--cache-dir", default="testgen_cache")
    tg.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "gui":
        from main import main as gui_main
        gui_main()
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "crack":
        cmd_crack(args)
    elif args.command == "testgen":
        cmd_testgen(args)


if __name__ == "__main__":
    main()
