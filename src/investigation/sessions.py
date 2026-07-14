"""ModelSession seam for the v3 loop (C7, M1 spec Part 3 + F8; M2 spec Part 4).

The loop never sees provider message formats. It supplies the semantic context
blocks produced by the C2 builder (an Anthropic-style messages list) and
consumes events. Per-provider sessions own the conversation shape and exploit
what each API allows.

M2 adds:
- ``AnthropicSession`` — native messages with a ``cache_control`` breakpoint at
  the C2 stable-prefix boundary (``capabilities.cache_breakpoints``). Anthropic
  has no credits this milestone, so it is exercised by fakes only.
- OpenAI within-episode server-state chaining (C7/F2): an episode-role
  ``OpenAISession`` on the Responses path sends the full context once with
  ``store=True``, then transmits only the unseen suffix with
  ``previous_response_id``. Any prefix mismatch or ``ModelProviderError`` falls
  back to a full stateless resend. The lead stays stateless.
- kind→session-factory routing (``session_factory`` for ``episode:<kind>``).

Budget entries are recorded per send; cost is never recomputed from run totals
(A7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent.model_provider import (
    AgentModelProvider,
    ModelProviderError,
    ModelResponse,
    _collect_assistant_blocks,
    _messages_to_openai_responses,
    _reasoning_passback_enabled,
    _requires_responses_api,
)
from investigation.state import BudgetEntry


@dataclass(frozen=True)
class SessionCapabilities:
    """What a session implementation supports (F8).

    Computed per (provider, MODEL): ``reasoning_passback`` is true only on the
    Responses path (gpt-5.6*), false for gpt-5.5. ``server_state`` is true only
    for an episode-role Responses session (C7). ``cache_breakpoints`` is true
    for AnthropicSession.
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


# ---------------------------------------------------------------------------
# Cache-hint handling (C2 stable-prefix breakpoint)
# ---------------------------------------------------------------------------
def _has_cache_hint(blocks: list[dict[str, Any]]) -> bool:
    for message in blocks:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("cache_hint") for b in content
        ):
            return True
    return False


def _strip_cache_hint(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the ``cache_hint`` marker from every block.

    ``build_lead_context`` marks the stable-prefix text block ``cache_hint:True``
    so an AnthropicSession can place a ``cache_control`` breakpoint; every other
    send path strips it so no provider sees an unknown field. Returns the SAME
    list object when there is nothing to strip (identity preserved for callers
    that assert ``messages is blocks``).
    """
    if not _has_cache_hint(blocks):
        return blocks
    out: list[dict[str, Any]] = []
    for message in blocks:
        content = message.get("content")
        if isinstance(content, list):
            out.append({**message, "content": [
                {k: v for k, v in b.items() if k != "cache_hint"}
                if isinstance(b, dict) else b
                for b in content
            ]})
        else:
            out.append(message)
    return out


def _apply_cache_control(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ``cache_hint`` markers to Anthropic ``cache_control`` breakpoints."""
    if not _has_cache_hint(blocks):
        return blocks
    out: list[dict[str, Any]] = []
    for message in blocks:
        content = message.get("content")
        if isinstance(content, list):
            new_content = []
            for b in content:
                if isinstance(b, dict) and b.get("cache_hint"):
                    nb = {k: v for k, v in b.items() if k != "cache_hint"}
                    nb["cache_control"] = {"type": "ephemeral"}
                    new_content.append(nb)
                else:
                    new_content.append(b)
            out.append({**message, "content": new_content})
        else:
            out.append(message)
    return out


class _BaseSession:
    """Shared plumbing: per-send budget entries and a native transcript."""

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

    def _record(self, response: ModelResponse) -> None:
        """Record one send's budget entry and native transcript exchange."""
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

    def send(
        self,
        blocks: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> ModelResponse:
        response = self._provider.send(
            messages=_strip_cache_hint(blocks),
            tools=tools,
            system=self._system,
            max_tokens=max_tokens,
        )
        self._record(response)
        return response

    def usage_entries(self) -> list[BudgetEntry]:
        return list(self._budget)

    def export_transcript(self) -> dict[str, Any]:
        # Part 7: export the system prompt once; each exchange records native
        # content blocks (Responses reasoning items ride along as provider_extra
        # blocks inside `response`) plus usage.
        return {
            "provider": self.provider_name,
            "model": self.model,
            "role": self._role,
            "system": self._system,
            "capabilities": {
                "server_state": self.capabilities.server_state,
                "reasoning_passback": self.capabilities.reasoning_passback,
                "cache_breakpoints": self.capabilities.cache_breakpoints,
                "strict_tools": self.capabilities.strict_tools,
            },
            "exchanges": list(self._exchanges),
        }


class GenericChatSession(_BaseSession):
    """Chat-completions providers (Ollama, OpenRouter)."""

    provider_tag = "generic_chat"

    def _compute_capabilities(self) -> SessionCapabilities:
        return SessionCapabilities(
            server_state=False,
            reasoning_passback=False,
            cache_breakpoints=False,
            strict_tools=False,
        )


class AnthropicSession(_BaseSession):
    """Anthropic provider — native messages with a stable-prefix cache breakpoint.

    Anthropic has no credits this milestone, so this session is exercised by
    fakes only (F12). It converts the C2 stable-prefix ``cache_hint`` marker into
    a ``cache_control`` breakpoint before sending; ``ClaudeAPI.send_message``
    already caches the system prompt and the last tool.
    """

    provider_tag = "anthropic"

    def _compute_capabilities(self) -> SessionCapabilities:
        return SessionCapabilities(
            server_state=False,
            reasoning_passback=False,
            cache_breakpoints=True,
            strict_tools=False,
        )

    def send(
        self,
        blocks: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> ModelResponse:
        response = self._provider.send(
            messages=_apply_cache_control(blocks),
            tools=tools,
            system=self._system,
            max_tokens=max_tokens,
        )
        self._record(response)
        return response


class OpenAISession(_BaseSession):
    """OpenAI provider — responses-native for gpt-5.6*, chat otherwise.

    For episode roles on the Responses path, uses within-episode server-state
    chaining (C7/F2): send the full context once with ``store=True``, then
    transmit only the unseen suffix with ``previous_response_id``. Any prefix
    mismatch or mid-episode ``ModelProviderError`` falls back to a full stateless
    resend (``store=False``) and continues stateless.
    """

    provider_tag = "openai"

    def __init__(
        self,
        provider: AgentModelProvider,
        system: str = "",
        *,
        role: str = "lead",
        category: str | None = None,
    ) -> None:
        super().__init__(provider, system, role=role, category=category)
        # Server-state chaining bookkeeping.
        self._prev_response_id: str | None = None
        # "Seen" = the input messages transmitted so far PLUS the assistant
        # message each send returned (F2). The next suffix is everything after.
        self._seen_blocks: list[dict[str, Any]] = []
        self._stateless_fallback: bool = False
        # For acceptance reporting: the responses-input-item count of the most
        # recent chained send's suffix.
        self.last_suffix_item_count: int | None = None
        self.last_response_id: str | None = None

    def _compute_capabilities(self) -> SessionCapabilities:
        responses_path = _requires_responses_api(self.model)
        server_state = responses_path and self._role.startswith("episode:")
        return SessionCapabilities(
            server_state=server_state,
            reasoning_passback=responses_path and _reasoning_passback_enabled(),
            cache_breakpoints=False,
            strict_tools=responses_path,
        )

    def send(
        self,
        blocks: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> ModelResponse:
        if not self.capabilities.server_state or self._stateless_fallback:
            return super().send(blocks, tools, max_tokens)

        blocks = _strip_cache_hint(list(blocks))
        seen = self._seen_blocks
        if not seen:
            suffix, prev_id = blocks, None
        elif seen == blocks[: len(seen)]:
            suffix, prev_id = blocks[len(seen):], self._prev_response_id
        else:
            # Prefix mismatch → the chain is broken; resend everything stateless.
            return self._resend_stateless(blocks, tools, max_tokens)

        try:
            response = self._provider.send(
                messages=suffix,
                tools=tools,
                system=self._system,
                max_tokens=max_tokens,
                store=True,
                previous_response_id=prev_id,
            )
        except ModelProviderError:
            return self._resend_stateless(blocks, tools, max_tokens)

        self._record(response)
        self.last_suffix_item_count = len(_messages_to_openai_responses(suffix))
        self._prev_response_id = getattr(getattr(response, "raw", None), "id", None)
        self.last_response_id = self._prev_response_id
        assistant_blocks, _tu, _txt = _collect_assistant_blocks(response)
        self._seen_blocks = blocks + [
            {"role": "assistant", "content": assistant_blocks}
        ]
        return response

    def _resend_stateless(
        self,
        blocks: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> ModelResponse:
        """Full stateless resend after a prefix mismatch / provider error (F2)."""
        self._stateless_fallback = True
        response = self._provider.send(
            messages=_strip_cache_hint(blocks),
            tools=tools,
            system=self._system,
            max_tokens=max_tokens,
            store=False,
            previous_response_id=None,
        )
        self._record(response)
        return response


# ---------------------------------------------------------------------------
# Session construction / registry
# ---------------------------------------------------------------------------
def _session_for_provider(
    provider: AgentModelProvider, system: str, role: str
) -> ModelSession:
    """Default routing by provider name (F6)."""
    name = getattr(provider, "provider_name", "")
    if name == "anthropic":
        return AnthropicSession(provider, system=system, role=role)
    if name == "openai":
        return OpenAISession(provider, system=system, role=role)
    return GenericChatSession(provider, system=system, role=role)


def make_lead_session(
    provider: AgentModelProvider, system: str = "", role: str = "lead"
) -> ModelSession:
    """Build the session for the lead role from a model provider."""
    return _session_for_provider(provider, system, role)


def default_session_builder(
    provider: AgentModelProvider, system: str = "", role: str = "lead"
) -> ModelSession:
    """Default builder used for episode roles with no registered fake (F6)."""
    return _session_for_provider(provider, system, role)


# Session factory registry. M1 shipped only "lead"; M2 adds default routing for
# any ``episode:<kind>`` role. Tests may register scripted fakes per role.
_SESSION_BUILDERS: dict[str, Any] = {
    "lead": make_lead_session,
}


def register_session_builder(role: str, builder: Any) -> None:
    _SESSION_BUILDERS[role] = builder


def session_factory(
    role_or_kind: str, provider: AgentModelProvider, system: str = ""
) -> ModelSession:
    """Create a ModelSession for the given role/kind.

    Lookup order (F6): exact registered builder → default provider routing for
    ``episode:<kind>`` roles → ValueError for any other unknown role.
    """
    builder = _SESSION_BUILDERS.get(role_or_kind)
    if builder is not None:
        return builder(provider, system, role_or_kind)
    if role_or_kind.startswith("episode:"):
        return default_session_builder(provider, system, role_or_kind)
    raise ValueError(f"No session builder registered for role: {role_or_kind!r}")
