"""Provider-neutral model interface for agent loops.

The current v2 loop still calls the Claude service directly, but newer loop
orchestration should reason about normalized model responses rather than
Anthropic-specific response objects.  This module is intentionally small:
provider adapters can add richer tracing later without changing the loop core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelUsage:
    """Token accounting normalized across model providers."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass(frozen=True)
class TextBlock:
    """One assistant text block."""

    text: str
    type: str = "text"


@dataclass(frozen=True)
class ToolUseBlock:
    """One assistant tool-use block."""

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    type: str = "tool_use"


ModelContentBlock = TextBlock | ToolUseBlock


@dataclass(frozen=True)
class ModelResponse:
    """A normalized assistant response used by Decipher's agent harness."""

    content: list[ModelContentBlock]
    usage: ModelUsage = field(default_factory=ModelUsage)
    raw: Any = None


class AgentModelProvider(Protocol):
    """Minimal model-provider surface needed by the agent harness."""

    model: str
    provider_name: str

    def send(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Send one model request and return a normalized response."""


def normalize_model_response(response: Any) -> ModelResponse:
    """Convert a provider response with Anthropic-like blocks to ModelResponse.

    This accepts both real Anthropic message objects and the SimpleNamespace
    fakes used by the test suite.  Adapters for other providers should return
    ModelResponse directly and bypass this helper.
    """

    if isinstance(response, ModelResponse):
        return response

    raw_usage = getattr(response, "usage", None)
    usage = ModelUsage(
        input_tokens=int(getattr(raw_usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw_usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(
            getattr(raw_usage, "cache_read_input_tokens", 0) or 0
        ),
    )

    content: list[ModelContentBlock] = []
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            content.append(TextBlock(text=str(getattr(block, "text", ""))))
        elif block_type == "tool_use":
            raw_input = getattr(block, "input", {}) or {}
            content.append(
                ToolUseBlock(
                    id=str(getattr(block, "id", "")),
                    name=str(getattr(block, "name", "")),
                    input=dict(raw_input) if isinstance(raw_input, dict) else {},
                )
            )

    return ModelResponse(content=content, usage=usage, raw=response)


class ClaudeModelProvider:
    """Adapter from the existing ClaudeAPI service to AgentModelProvider."""

    provider_name = "anthropic"

    def __init__(self, api: Any) -> None:
        self.api = api
        self.model = api.model

    def send(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 4096,
    ) -> ModelResponse:
        response = self.api.send_message(
            messages=messages,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
        )
        return normalize_model_response(response)
