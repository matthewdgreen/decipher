"""One-shot + Python-code baseline runner.

Gives Claude a single run_python tool and asks it to decrypt the cipher
using whatever code it wants to write.  No domain-specific cryptanalysis
tools, no workspace — just Claude's own reasoning plus a code executor.

Used to compare against the full agentic framework.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from benchmark.loader import TestData
from benchmark.scorer import score_decryption
from services.claude_api import ClaudeAPI, ClaudeAPIError


@dataclass
class BaselineResult:
    test_id: str
    char_accuracy: float
    word_accuracy: float
    iterations: int
    elapsed: float
    decryption: str
    ground_truth: str
    error: str = ""
    tool_calls_log: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # full conversation history
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model: str = ""
    artifact_path: str = ""

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "test_id": self.test_id,
            "model": self.model,
            "runner": "baseline_one_shot",
            "char_accuracy": self.char_accuracy,
            "word_accuracy": self.word_accuracy,
            "iterations": self.iterations,
            "elapsed": round(self.elapsed, 1),
            "decryption": self.decryption,
            "ground_truth": self.ground_truth,
            "error": self.error,
            "tool_calls_log": self.tool_calls_log,
            "messages": self.messages,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


_TOOL_DEFINITIONS = [
    {
        "name": "run_python",
        "description": (
            "Execute Python code and return its stdout/stderr. "
            "Use freely for frequency analysis, substitution testing, "
            "n-gram scoring, dictionary lookups, or anything else helpful. "
            "Print results to stdout. stdlib only; no external packages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python 3 code to execute. Print outputs to stdout.",
                }
            },
            "required": ["code"],
        },
    }
]

_SYSTEM_TMPL = """\
You are an expert cryptanalyst. Decrypt the substitution cipher below and \
recover the original {language} plaintext.

You have a `run_python` tool — use it as much as you need to analyse \
frequencies, test substitutions, score candidates, and refine your answer.

{format_note}

When you are confident, output your final answer on its own line as:
SOLUTION: <UPPERCASE PLAINTEXT>"""

_FORMAT_WORD_BOUNDARY = (
    "The plaintext has WORD BOUNDARIES — output space-separated words, "
    "e.g. SOLUTION: THE QUICK BROWN FOX"
)
_FORMAT_NO_BOUNDARY = (
    "The plaintext has NO WORD BOUNDARIES — output a single run of letters "
    "with no spaces, e.g. SOLUTION: THEQUICKBROWNFOX"
)


def run_baseline(
    test_data: TestData,
    api: ClaudeAPI,
    language: str = "en",
    max_iterations: int = 10,
    verbose: bool = False,
    artifact_dir: str | Path | None = None,
) -> BaselineResult:
    """Run the one-shot + code baseline on a single test case."""
    ground_truth = test_data.plaintext
    # Detect word boundaries from the cipher structure, not from the plaintext.
    # The canonical transcription uses " | " as word separator when words are
    # delimited; this is observable from the ciphertext alone.
    word_boundaries = " | " in test_data.canonical_transcription
    format_note = _FORMAT_WORD_BOUNDARY if word_boundaries else _FORMAT_NO_BOUNDARY

    system = _SYSTEM_TMPL.format(language=language, format_note=format_note)

    user_msg = (
        f"Cipher text ({test_data.test.cipher_system}, "
        f"{'word-boundary' if word_boundaries else 'no-boundary'}):\n\n"
        f"{test_data.canonical_transcription}\n\n"
        "Decrypt this cipher. Use run_python as needed, then give your SOLUTION."
    )

    messages: list[dict] = [{"role": "user", "content": user_msg}]
    tool_calls_log: list[dict] = []
    final_text = ""
    error = ""
    t0 = time.time()

    try:
        for iteration in range(max_iterations):
            response = api.send_message(
                messages=messages,
                tools=_TOOL_DEFINITIONS,
                system=system,
                max_tokens=8192,
            )

            assistant_content = []
            tool_uses = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                    final_text = block.text  # keep the most recent text block
                    if verbose:
                        print(f"    [baseline] {block.text[:200]}")
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                    tool_uses.append(block)

            messages.append({"role": "assistant", "content": assistant_content})

            if not tool_uses:
                # Claude is done
                break

            # Execute each tool call
            tool_results = []
            for tu in tool_uses:
                if tu.name == "run_python":
                    code = tu.input.get("code", "")
                    output = _execute_python(code)
                    if verbose:
                        print(f"    [run_python] → {output[:120]}")
                    tool_calls_log.append({"iteration": iteration + 1, "code": code, "output": output})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": output,
                    })
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": f"Unknown tool: {tu.name}",
                    })

            messages.append({"role": "user", "content": tool_results})

    except ClaudeAPIError as e:
        error = str(e)

    elapsed = time.time() - t0
    decryption = _extract_solution(final_text)
    iterations = len([m for m in messages if m["role"] == "assistant"])

    score = score_decryption(
        test_id=test_data.test.test_id,
        decrypted=decryption,
        ground_truth=ground_truth,
        agent_score=0.0,
        status="solved" if decryption else "failed",
    )

    result = BaselineResult(
        test_id=test_data.test.test_id,
        char_accuracy=score.char_accuracy,
        word_accuracy=score.word_accuracy,
        iterations=iterations,
        elapsed=elapsed,
        decryption=decryption,
        ground_truth=ground_truth,
        error=error,
        tool_calls_log=tool_calls_log,
        messages=messages,
        model=api.model,
    )

    if artifact_dir is not None:
        artifact_path = Path(artifact_dir) / result.test_id / f"baseline_{result.run_id}.json"
        try:
            result.save(artifact_path)
            result.artifact_path = str(artifact_path)
        except Exception:  # noqa: BLE001
            pass

    return result


def _execute_python(code: str, timeout: int = 15) -> str:
    """Run Python code in a subprocess; return stdout + stderr, truncated."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + "STDERR:\n" + result.stderr
        return output[:3000] or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


def _extract_solution(text: str) -> str:
    """Pull the SOLUTION: line from Claude's final response."""
    m = re.search(r"SOLUTION\s*:\s*(.+)", text, re.IGNORECASE)
    if not m:
        return ""
    raw = m.group(1).split("\n")[0].strip()
    # Keep only letters and spaces; uppercase
    cleaned = re.sub(r"[^A-Za-z ]", "", raw).upper()
    return " ".join(cleaned.split())  # normalise whitespace
