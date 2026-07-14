"""ModelSession seam for the v3 loop (C7, M1 spec Part 3 + F8).

The loop never sees provider message formats. It supplies the semantic context
blocks produced by the C2 builder (an Anthropic-style messages list) and
consumes events. Per-provider sessions own the conversation shape and exploit
what each API allows.

M1 ships two sessions:
- ``OpenAISession`` wraps ``OpenAIModelProvider`` — responses-native for
  gpt-5.6* (with stateless encrypted-reasoning passback reused from the landed
  provider_extra mechanics), chat completions otherwise (gpt-5.5). It does NOT
  duplicate the converters — the underlying provider owns them.
- ``GenericChatSession`` wraps the chat-completions providers (Anthropic via
  ClaudeModelProvider, Ollama, OpenRouter) — the neutral behavior as one
  implementation.

Server-side chaining (``previous_response_id``) and AnthropicSession cache
breakpoints are M2; M1 sessions are stateless. Budget entries are recorded per
send; cost is never recomputed from run totals (A7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent.model_provider import (
    AgentModelProvider,
    ModelResponse,
    _collect_assistant_blocks,
    _reasoning_passback_enabled,
    _requires_responses_api,
)
from investigation.state import BudgetEntry


@dataclass(frozen=True)
class SessionCapabilities:
    """What a session implementation supports (F8).

    Computed per (provider, MODEL): ``reasoning_passback`` is true only on the
    Responses path (gpt-5.6*), false for gpt-5.5.
    """

    server_state: bool = False
    reasoning_passback: bool = False
    cache_breakpoints: bool = False
    strict_tools: bool = False


class ModelSession(Protocol):
    """Minimal seam the v3 loop needs from a live model context."""

    capabilities: SessionCapabilities

    def send(
        self,
        blocks: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> ModelResponse:
        """Send one request. ``blocks`` is the complete logical context for the
        call (F8: a server_state-capable session may later transmit only the
        unseen suffix — additive, no protocol change)."""

    def usage_entries(self) -> list[BudgetEntry]:
        """Return all budget entries recorded so far (one per send)."""

    def export_transcript(self) -> dict[str, Any]:
        """Return a provider-tagged native transcript for the artifact."""


class _BaseSession:
    """Shared plumbing: per-send budget entries and a modest transcript."""

    provider_tag = "generic"

    def __init__(
        self,
        provider: AgentModelProvider,
        system: str = "",
        *,
        role: str = "lead",
        category: str | None = None,
    ) -> None:
        self._provider = provider
        self._system = system
        self._role = role
        self._category = category or role
        self._budget: list[BudgetEntry] = []
        self._exchanges: list[dict[str, Any]] = []
        self.capabilities = self._compute_capabilities()

    # Subclasses override.
    def _compute_capabilities(self) -> SessionCapabilities:
        return SessionCapabilities()

    @property
    def model(self) -> str:
        return getattr(self._provider, "model", "")

    @property
    def provider_name(self) -> str:
        return getattr(self._provider, "provider_name", self.provider_tag)

    def send(
        self,
        blocks: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> ModelResponse:
        response = self._provider.send(
            messages=blocks,
            tools=tools,
            system=self._system,
            max_tokens=max_tokens,
        )
        usage = response.usage
        self._budget.append(
            BudgetEntry(
                category=self._category,
                provider=self.provider_name,
                model=self.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_input_tokens,
            )
        )
        assistant_blocks, _tool_uses, _text = _collect_assistant_blocks(response)
        self._exchanges.append({
            "response": assistant_blocks,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_input_tokens,
            },
        })
        return response

    def usage_entries(self) -> list[BudgetEntry]:
        return list(self._budget)

    def export_transcript(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model,
            "role": self._role,
            "capabilities": {
                "server_state": self.capabilities.server_state,
                "reasoning_passback": self.capabilities.reasoning_passback,
                "cache_breakpoints": self.capabilities.cache_breakpoints,
                "strict_tools": self.capabilities.strict_tools,
            },
            "exchanges": list(self._exchanges),
        }


class GenericChatSession(_BaseSession):
    """Chat-completions providers (Anthropic, Ollama, OpenRouter)."""

    provider_tag = "generic_chat"

    def _compute_capabilities(self) -> SessionCapabilities:
        # Neutral behavior: no reasoning passback, no server state. M1 does not
        # place Anthropic cache breakpoints (AnthropicSession lands with M2).
        return SessionCapabilities(
            server_state=False,
            reasoning_passback=False,
            cache_breakpoints=False,
            strict_tools=False,
        )


class OpenAISession(_BaseSession):
    """OpenAI provider — responses-native for gpt-5.6*, chat otherwise."""

    provider_tag = "openai"

    def _compute_capabilities(self) -> SessionCapabilities:
        responses_path = _requires_responses_api(self.model)
        return SessionCapabilities(
            server_state=False,  # previous_response_id chaining is M2
            reasoning_passback=responses_path and _reasoning_passback_enabled(),
            cache_breakpoints=False,
            strict_tools=responses_path,
        )


def make_lead_session(
    provider: AgentModelProvider, system: str = "", role: str = "lead"
) -> ModelSession:
    """Build the session for the lead role from a model provider."""
    if getattr(provider, "provider_name", "") == "openai":
        return OpenAISession(provider, system=system, role=role)
    return GenericChatSession(provider, system=system, role=role)


# Session factory registry (M1 has a single role: "lead"). Tests may register
# fakes per provider shape.
_SESSION_BUILDERS: dict[str, Any] = {
    "lead": make_lead_session,
}


def register_session_builder(role: str, builder: Any) -> None:
    _SESSION_BUILDERS[role] = builder


def session_factory(
    role_or_kind: str, provider: AgentModelProvider, system: str = ""
) -> ModelSession:
    """Create a ModelSession for the given role/kind. M1: only 'lead'."""
    builder = _SESSION_BUILDERS.get(role_or_kind)
    if builder is None:
        raise ValueError(f"No session builder registered for role: {role_or_kind!r}")
    return builder(provider, system, role_or_kind)
