from __future__ import annotations

import os
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from agent.prompts import get_system_prompt, initial_context
from agent.state import AgentState
from agent.tools import TOOL_DEFINITIONS, ToolExecutor
from analysis import dictionary, frequency, ic, pattern
from models.session import Session
from services.claude_api import ClaudeAPI, ClaudeAPIError
from services.settings import Settings


class AgentWorker(QObject):
    """Runs the agentic cracking loop in a background thread."""

    iteration_complete = Signal(int, str)  # iteration number, summary
    text_delta = Signal(str)  # streaming text from Claude
    tool_called = Signal(str, str)  # tool name, result preview
    progress = Signal(float)  # best score so far
    finished = Signal(str)  # final status
    error = Signal(str)  # error message

    def __init__(
        self,
        session: Session,
        claude_api: ClaudeAPI,
        settings: Settings,
        language: str = "en",
    ) -> None:
        super().__init__()
        self.session = session
        self.api = claude_api
        self.settings = settings
        self.language = language
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self._run_loop()
        except ClaudeAPIError as e:
            self.error.emit(f"API Error: {e}")
        except Exception as e:
            self.error.emit(f"Unexpected error: {e}")

    def _run_loop(self) -> None:
        ct = self.session.cipher_text
        if ct is None:
            self.error.emit("No ciphertext loaded")
            return

        # Load dictionary resources for the target language
        dict_path = dictionary.get_dictionary_path(self.language)
        word_set = dictionary.load_word_set(dict_path)
        word_list = pattern.load_word_list(dict_path)
        pattern_dict = pattern.build_pattern_dictionary(word_list)

        state = AgentState(max_iterations=self.settings.max_iterations)

        # Create tool executor with state for rollback support
        executor = ToolExecutor(self.session, word_set, pattern_dict, agent_state=state)

        # Compute initial analysis
        freq_data = frequency.sorted_frequency(ct.tokens)
        freq_display = [
            (ct.alphabet.symbol_for(tid), count,
             count / len(ct.tokens) * 100)
            for tid, count in freq_data
        ]
        ic_value = ic.index_of_coincidence(ct.tokens, ct.alphabet.size)

        # Score the raw text BEFORE any mapping to detect plaintext/trivial ciphers
        # Use the raw text (with punctuation) so the agent can read it naturally
        raw_for_display = ct.raw
        pre_mapping_score = dictionary.score_plaintext(raw_for_display.upper(), word_set)

        # Build initial message
        context = initial_context(
            cipher_display=raw_for_display,
            alphabet_symbols=ct.alphabet.symbols,
            freq_data=freq_display,
            ic_value=ic_value,
            total_tokens=len(ct.tokens),
            pre_mapping_score=pre_mapping_score,
            language=self.language,
        )

        state.status = "running"
        state.add_user_message(context)

        for iteration in range(state.max_iterations):
            if self._stop_requested:
                state.status = "stopped"
                break

            state.iteration = iteration + 1

            # Checkpoint before each iteration
            pre_score = dictionary.score_plaintext(
                self.session.apply_key(), word_set
            )
            state.save_checkpoint(pre_score, self.session.key)
            state.previous_score = pre_score

            # Call Claude with tools
            response = self.api.send_message(
                messages=state.messages,
                tools=TOOL_DEFINITIONS,
                system=get_system_prompt(self.language),
            )

            # Process response content blocks
            assistant_content: list[dict[str, Any]] = []
            tool_uses: list[dict[str, Any]] = []
            text_parts: list[str] = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({
                        "type": "text",
                        "text": block.text,
                    })
                    text_parts.append(block.text)
                    self.text_delta.emit(block.text)
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

            # Execute tool calls
            if tool_uses:
                tool_results: list[tuple[str, str]] = []
                for tool in tool_uses:
                    result = executor.execute(tool["name"], tool["input"])
                    tool_results.append((tool["id"], result))
                    preview = result[:200] + "..." if len(result) > 200 else result
                    self.tool_called.emit(tool["name"], preview)

                # Check current score before sending results to Claude
                decrypted = self.session.apply_key()
                score = dictionary.score_plaintext(decrypted, word_set)
                state.update_best(score, self.session.key)
                self.progress.emit(score)

                # If score dropped, inject a warning into the tool results
                if state.score_dropped(score):
                    drop_warning = (
                        f"\n⚠️ WARNING: Score DROPPED from {state.previous_score:.2%} "
                        f"to {score:.2%} after your last changes. "
                        f"Your changes made things WORSE. "
                        f"Call rollback to undo and try a different approach."
                    )
                    # Append warning as text to the last tool result
                    last_id, last_result = tool_results[-1]
                    tool_results[-1] = (last_id, last_result + drop_warning)

                state.add_tool_results(tool_results)

                summary = (
                    f"Iteration {state.iteration}: "
                    f"{len(tool_uses)} tool calls, score={score:.2%}"
                )
                if state.score_dropped(score):
                    summary += f" (DROPPED from {state.previous_score:.2%}!)"
                self.iteration_complete.emit(state.iteration, summary)

                # Only auto-stop at very high confidence AND all symbols mapped
                if score >= 0.98 and self.session.is_complete:
                    state.status = "solved"
                    break
            else:
                # No tool calls — agent is done or stuck
                summary = f"Iteration {state.iteration}: no tool calls (agent finished)"
                self.iteration_complete.emit(state.iteration, summary)
                if state.best_score >= 0.5:
                    state.status = "solved"
                else:
                    state.status = "stuck"
                break

        if state.status == "running":
            state.status = "stuck"

        # Restore best key if we overshot
        if state.best_score > dictionary.score_plaintext(
            self.session.apply_key(), word_set
        ):
            self.session.set_full_key(state.best_key)

        self.finished.emit(state.status)


class AgentLoop:
    """Manages the agent worker thread lifecycle."""

    def __init__(
        self,
        session: Session,
        claude_api: ClaudeAPI,
        settings: Settings,
        language: str = "en",
    ) -> None:
        self.session = session
        self.api = claude_api
        self.settings = settings
        self.language = language
        self._thread: QThread | None = None
        self._worker: AgentWorker | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self) -> AgentWorker:
        """Start the agent loop in a background thread. Returns the worker for signal connections."""
        if self.is_running:
            raise RuntimeError("Agent is already running")

        self._thread = QThread()
        self._worker = AgentWorker(self.session, self.api, self.settings, language=self.language)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()
        return self._worker

    def stop(self) -> None:
        if self._worker:
            self._worker.stop()

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
