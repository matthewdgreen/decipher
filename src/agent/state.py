from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Checkpoint:
    """Snapshot of the key state at a point in time."""
    iteration: int
    score: float
    key: dict[int, int]
    label: str = ""


class AgentState:
    """Tracks the state of an agentic cracking session."""

    def __init__(self, max_iterations: int = 25) -> None:
        self.messages: list[dict[str, Any]] = []
        self.iteration: int = 0
        self.max_iterations: int = max_iterations
        self.best_score: float = 0.0
        self.best_key: dict[int, int] = {}
        self.status: str = "idle"  # idle, running, solved, stuck, stopped
        self.checkpoints: list[Checkpoint] = []
        self.previous_score: float = 0.0

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_use_id: str, result: str) -> None:
        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                }
            ],
        })

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        """Add multiple tool results in a single user message."""
        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                }
                for tool_use_id, result in results
            ],
        })

    def update_best(self, score: float, key: dict[int, int]) -> None:
        if score > self.best_score:
            self.best_score = score
            self.best_key = dict(key)

    def save_checkpoint(self, score: float, key: dict[int, int], label: str = "") -> None:
        self.checkpoints.append(Checkpoint(
            iteration=self.iteration,
            score=score,
            key=dict(key),
            label=label or f"iteration_{self.iteration}",
        ))

    def get_checkpoint(self, index: int = -1) -> Checkpoint | None:
        if not self.checkpoints:
            return None
        try:
            return self.checkpoints[index]
        except IndexError:
            return None

    def score_dropped(self, current_score: float) -> bool:
        """Check if score decreased significantly from previous iteration."""
        return current_score < self.previous_score - 0.02
