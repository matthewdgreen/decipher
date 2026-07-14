"""ModelSession seam tests (M1 Part 3/5): routing, budgets, transcript, caps."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.model_provider import (
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolUseBlock,
    _reasoning_passback_enabled,
)
from investigation.sessions import (
    GenericChatSession,
    OpenAISession,
    make_lead_session,
    register_session_builder,
    session_factory,
)


class _FakeProvider:
    def __init__(self, model: str, provider_name: str) -> None:
        self.model = model
        self.provider_name = provider_name
        self.calls: list[dict] = []

    def send(self, *, messages, tools=None, system="", max_tokens=4096) -> ModelResponse:
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return ModelResponse(
            content=[
                TextBlock(text="thinking"),
                ToolUseBlock(id="a1", name="decode_show", input={"branch": "main"}),
            ],
            usage=ModelUsage(input_tokens=1200, output_tokens=80,
                             cache_read_input_tokens=300),
        )


def test_make_lead_session_routes_by_provider():
    assert isinstance(
        make_lead_session(_FakeProvider("gpt-5.5", "openai")), OpenAISession
    )
    assert isinstance(
        make_lead_session(_FakeProvider("claude-sonnet-4-6", "anthropic")),
        GenericChatSession,
    )
    assert isinstance(
        make_lead_session(_FakeProvider("qwen3:14b", "ollama")), GenericChatSession
    )


def test_openai_session_capabilities_depend_on_model():
    chat = OpenAISession(_FakeProvider("gpt-5.5", "openai"))
    assert chat.capabilities.reasoning_passback is False
    assert chat.capabilities.strict_tools is False
    assert chat.capabilities.server_state is False

    responses = OpenAISession(_FakeProvider("gpt-5.6-sol", "openai"))
    assert responses.capabilities.strict_tools is True
    # reasoning_passback is true on the Responses path when the env flag is on.
    assert responses.capabilities.reasoning_passback is _reasoning_passback_enabled()


def test_generic_chat_session_capabilities_are_neutral():
    session = GenericChatSession(_FakeProvider("claude-sonnet-4-6", "anthropic"))
    assert session.capabilities.reasoning_passback is False
    assert session.capabilities.server_state is False
    assert session.capabilities.cache_breakpoints is False
    assert session.capabilities.strict_tools is False


def test_send_forwards_blocks_and_records_budget_entries():
    provider = _FakeProvider("gpt-5.5", "openai")
    session = make_lead_session(provider, system="be brief")
    blocks = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    tools = [{"name": "decode_show", "description": "", "input_schema": {}}]

    session.send(blocks, tools=tools, max_tokens=2048)
    session.send(blocks, tools=tools, max_tokens=2048)

    # blocks and system are forwarded to the underlying provider verbatim.
    assert provider.calls[0]["messages"] is blocks
    assert provider.calls[0]["system"] == "be brief"

    entries = session.usage_entries()
    assert len(entries) == 2
    entry = entries[0]
    assert entry.category == "lead"
    assert entry.provider == "openai"
    assert entry.model == "gpt-5.5"
    assert entry.input_tokens == 1200
    assert entry.output_tokens == 80
    assert entry.cache_read_tokens == 300


def test_export_transcript_is_provider_tagged_and_nonempty():
    provider = _FakeProvider("gpt-5.6-sol", "openai")
    session = make_lead_session(provider)
    session.send([{"role": "user", "content": [{"type": "text", "text": "go"}]}])
    transcript = session.export_transcript()
    assert transcript["provider"] == "openai"
    assert transcript["model"] == "gpt-5.6-sol"
    assert transcript["exchanges"]
    # The assistant blocks (text + tool_use) are captured for the artifact.
    blocks = transcript["exchanges"][0]["response"]
    assert any(b.get("type") == "tool_use" for b in blocks)


def test_session_factory_registry():
    provider = _FakeProvider("gpt-5.5", "openai")
    session = session_factory("lead", provider, "sys")
    assert isinstance(session, OpenAISession)

    with pytest.raises(ValueError):
        session_factory("unknown_role", provider, "sys")

    marker = object()
    register_session_builder("probe", lambda p, s, r: marker)
    try:
        assert session_factory("probe", provider, "sys") is marker
    finally:
        # keep the registry clean for other tests
        from investigation.sessions import _SESSION_BUILDERS
        _SESSION_BUILDERS.pop("probe", None)
