from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from agent.prompts import get_system_prompt, initial_context
from agent.state import AgentState
from agent.tools import TOOL_DEFINITIONS, ToolExecutor
from analysis import dictionary, frequency, ic, pattern
from benchmark.loader import TestData
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from models.session import Session
from services.claude_api import ClaudeAPI, ClaudeAPIError


@dataclass
class RunResult:
    """Result of running the agent on a single benchmark test."""

    test_id: str
    status: str  # solved, stuck, stopped, error
    final_decryption: str
    best_score: float
    iterations_used: int
    elapsed_seconds: float
    error_message: str = ""
    iteration_log: list[str] = field(default_factory=list)


def parse_canonical_transcription(canonical_text: str) -> CipherText:
    """Parse a canonical transcription (space-separated S-tokens, | word separators)
    into a CipherText object.

    Format: S025 S012 S006 | S003 S007 S012 S019 | S005 S009 S010 S009
    """
    # Join all lines — newlines are word boundaries too
    lines = canonical_text.strip().split("\n")
    full = " | ".join(line.strip() for line in lines if line.strip())

    # Split into words on " | "
    word_strings = full.split(" | ")

    # Collect all unique S-tokens in order of appearance
    seen: set[str] = set()
    symbols: list[str] = []
    for word_str in word_strings:
        for token in word_str.split():
            if token not in seen:
                seen.add(token)
                symbols.append(token)

    alphabet = Alphabet(symbols)

    # Reconstruct raw text with " | " as word separator
    raw = " | ".join(word_strings)

    return CipherText(raw=raw, alphabet=alphabet, source="benchmark", separator=" | ")


class BenchmarkRunner:
    """Runs the agent on benchmark tests headlessly."""

    def __init__(
        self,
        claude_api: ClaudeAPI,
        max_iterations: int = 25,
        verbose: bool = False,
        language: str | None = None,
    ) -> None:
        self.api = claude_api
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.default_language = language  # None = auto-detect from test metadata
        self._dict_cache: dict[str, tuple[set[str], dict[str, list[str]]]] = {}

    def _get_dict_resources(self, language: str) -> tuple[set[str], dict[str, list[str]]]:
        """Load and cache dictionary resources for a language."""
        if language not in self._dict_cache:
            dict_path = dictionary.get_dictionary_path(language)
            word_set = dictionary.load_word_set(dict_path)
            word_list = pattern.load_word_list(dict_path)
            pattern_dict = pattern.build_pattern_dictionary(word_list)
            self._dict_cache[language] = (word_set, pattern_dict)
        return self._dict_cache[language]

    def run_test(self, test_data: TestData, language: str | None = None) -> RunResult:
        """Run the agent on a single benchmark test and return the result.

        Language is resolved in priority order:
        1. Explicit language argument
        2. Runner's default_language
        3. Auto-detect from test metadata (plaintext_language field)
        4. Fall back to English
        """
        test_id = test_data.test.test_id
        start_time = time.time()
        log: list[str] = []

        # Resolve language
        lang = language or self.default_language
        if lang is None and test_data.symbol_map:
            # Try to infer from source name
            source = test_data.test.test_id.split("_")[0]
            lang_map = {"borg": "la", "copiale": "de"}
            lang = lang_map.get(source)
        if lang is None:
            lang = "en"

        def log_msg(msg: str) -> None:
            log.append(msg)
            if self.verbose:
                print(f"  [{test_id}] {msg}")

        try:
            # Parse canonical transcription into CipherText
            ct = parse_canonical_transcription(test_data.canonical_transcription)
            word_set, pattern_dict = self._get_dict_resources(lang)
            log_msg(f"Parsed ciphertext: {ct.alphabet.size} symbols, {len(ct.tokens)} tokens, {len(ct.words)} words (lang={lang})")

            # Set up session
            session = Session()
            session.set_cipher_text(ct)

            # Set up agent state
            state = AgentState(max_iterations=self.max_iterations)
            executor = ToolExecutor(session, word_set, pattern_dict, agent_state=state)

            # Compute initial analysis
            freq_data = frequency.sorted_frequency(ct.tokens)
            freq_display = [
                (ct.alphabet.symbol_for(tid), count,
                 count / len(ct.tokens) * 100)
                for tid, count in freq_data
            ]
            ic_value = ic.index_of_coincidence(ct.tokens, ct.alphabet.size)

            # Pre-mapping score (will be low for canonical S-tokens)
            raw_for_display = ct.raw
            pre_mapping_score = dictionary.score_plaintext(raw_for_display.upper(), word_set)

            # Build initial context
            context = initial_context(
                cipher_display=raw_for_display,
                alphabet_symbols=ct.alphabet.symbols,
                freq_data=freq_display,
                ic_value=ic_value,
                total_tokens=len(ct.tokens),
                pre_mapping_score=pre_mapping_score,
                language=lang,
            )

            state.status = "running"
            state.add_user_message(context)

            # Run the agent loop
            for iteration in range(state.max_iterations):
                state.iteration = iteration + 1

                # Checkpoint
                pre_score = dictionary.score_plaintext(
                    session.apply_key(), word_set
                )
                state.save_checkpoint(pre_score, session.key)
                state.previous_score = pre_score

                log_msg(f"Iteration {state.iteration}, pre-score={pre_score:.2%}")

                # Call Claude (print inline so user sees activity)
                if not self.verbose:
                    print(f"  iter {state.iteration}...", end="", flush=True)
                try:
                    response = self.api.send_message(
                        messages=state.messages,
                        tools=TOOL_DEFINITIONS,
                        system=get_system_prompt(lang),
                    )
                except ClaudeAPIError as e:
                    log_msg(f"API error: {e}")
                    if not self.verbose:
                        print(f" ERROR: {e}", flush=True)
                    state.status = "error"
                    break

                # Process response
                assistant_content: list[dict[str, Any]] = []
                tool_uses: list[dict[str, Any]] = []

                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({
                            "type": "text",
                            "text": block.text,
                        })
                        if self.verbose:
                            # Print agent's reasoning (truncated)
                            text_preview = block.text[:200]
                            if len(block.text) > 200:
                                text_preview += "..."
                            log_msg(f"Agent: {text_preview}")
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                        tool_uses.append({
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

                state.add_assistant_message(assistant_content)

                if tool_uses:
                    tool_results: list[tuple[str, str]] = []
                    for tool in tool_uses:
                        result = executor.execute(tool["name"], tool["input"])
                        tool_results.append((tool["id"], result))
                        log_msg(f"Tool: {tool['name']}")

                    # Score
                    decrypted = session.apply_key()
                    score = dictionary.score_plaintext(decrypted, word_set)
                    state.update_best(score, session.key)

                    # Inject warning on score drop
                    if state.score_dropped(score):
                        drop_warning = (
                            f"\n⚠️ WARNING: Score DROPPED from {state.previous_score:.2%} "
                            f"to {score:.2%} after your last changes. "
                            f"Your changes made things WORSE. "
                            f"Call rollback to undo and try a different approach."
                        )
                        last_id, last_result = tool_results[-1]
                        tool_results[-1] = (last_id, last_result + drop_warning)

                    state.add_tool_results(tool_results)

                    log_msg(f"Score: {score:.2%} (best: {state.best_score:.2%})")
                    if not self.verbose:
                        print(f" score={score:.2%}", flush=True)

                    if score >= 0.98 and session.is_complete:
                        state.status = "solved"
                        break

                    # Detect stuckness: same score for 4+ iterations
                    if len(state.checkpoints) >= 4:
                        recent = [cp.score for cp in state.checkpoints[-4:]]
                        if len(set(f"{s:.4f}" for s in recent)) == 1:
                            log_msg("Stuck: score unchanged for 4 iterations, stopping")
                            if not self.verbose:
                                print("  (stuck, stopping early)", flush=True)
                            state.status = "stuck"
                            break
                else:
                    log_msg("No tool calls — agent finished")
                    if not self.verbose:
                        print(" done", flush=True)
                    state.status = "solved" if state.best_score >= 0.5 else "stuck"
                    break

            if state.status == "running":
                state.status = "stuck"

            # Restore best key if we overshot
            if state.best_score > dictionary.score_plaintext(
                session.apply_key(), word_set
            ):
                session.set_full_key(state.best_key)

            final_decryption = session.apply_key()
            elapsed = time.time() - start_time

            return RunResult(
                test_id=test_id,
                status=state.status,
                final_decryption=final_decryption,
                best_score=state.best_score,
                iterations_used=state.iteration,
                elapsed_seconds=elapsed,
                iteration_log=log,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return RunResult(
                test_id=test_id,
                status="error",
                final_decryption="",
                best_score=0.0,
                iterations_used=0,
                elapsed_seconds=elapsed,
                error_message=str(e),
                iteration_log=log,
            )
