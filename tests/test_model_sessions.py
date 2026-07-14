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
    _messages_to_openai_chat,
    _reasoning_passback_enabled,
)
from investigation.sessions import (
    AnthropicSession,
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
    # M2: anthropic routes to AnthropicSession (cache breakpoints).
    assert isinstance(
        make_lead_session(_FakeProvider("claude-sonnet-4-6", "anthropic")),
        AnthropicSession,
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


class _FakeResponsesProvider:
    """OpenAI-shaped provider recording server-state kwargs on each send.

    ``fail_attempts`` is a set of 1-based attempt numbers that raise
    ModelProviderError instead of answering (the attempt is still recorded so
    tests can index every call).
    """

    provider_name = "openai"

    def __init__(self, model="gpt-5.6-luna", fail_attempts=()):
        self.model = model
        self.calls: list[dict] = []
        self._n = 0
        self.fail_attempts = set(fail_attempts)

    def send(self, *, messages, tools=None, system="", max_tokens=4096,
             previous_response_id=None, store=None):
        self._n += 1
        self.calls.append({
            "messages": messages, "store": store,
            "previous_response_id": previous_response_id,
        })
        if self._n in self.fail_attempts:
            from agent.model_provider import ModelProviderError
            raise ModelProviderError("simulated responses outage")
        from types import SimpleNamespace
        raw = SimpleNamespace(id=f"resp_{self._n}")
        return ModelResponse(
            content=[ToolUseBlock(id=f"t{self._n}", name="decode_show", input={})],
            usage=ModelUsage(input_tokens=10, output_tokens=5,
                             cache_read_input_tokens=0),
            raw=raw,
        )


def test_episode_session_has_server_state_capability():
    lead = OpenAISession(_FakeResponsesProvider("gpt-5.6-luna"), role="lead")
    assert lead.capabilities.server_state is False  # lead stays stateless
    survey = OpenAISession(_FakeResponsesProvider("gpt-5.6-luna"),
                           role="episode:survey")
    assert survey.capabilities.server_state is True
    # gpt-5.5 (chat path) never gets server state, even for episode roles.
    chat = OpenAISession(_FakeResponsesProvider("gpt-5.5"), role="episode:survey")
    assert chat.capabilities.server_state is False


def test_openai_server_state_second_send_is_suffix_only():
    provider = _FakeResponsesProvider("gpt-5.6-luna")
    sess = OpenAISession(provider, role="episode:survey")
    assert sess.capabilities.server_state is True

    b1 = [{"role": "user", "content": [{"type": "text", "text": "ctx"}]}]
    sess.send(b1)
    # Send 1 transmitted everything with store=True and no previous id.
    assert provider.calls[0]["store"] is True
    assert provider.calls[0]["previous_response_id"] is None
    assert provider.calls[0]["messages"] == b1

    # The loop appends the returned assistant message + a tool_result and resends.
    assistant = {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "decode_show", "input": {}}]}
    tool_result = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]}
    b2 = b1 + [assistant, tool_result]
    sess.send(b2)

    # Send 2 transmits ONLY the unseen suffix (the tool_result) with the prior id.
    assert provider.calls[1]["store"] is True
    assert provider.calls[1]["previous_response_id"] == "resp_1"
    assert provider.calls[1]["messages"] == [tool_result]
    assert sess.last_suffix_item_count == 1


def test_openai_server_state_provider_error_falls_back_and_stays_stateless():
    """F2: a mid-episode ModelProviderError on a chained send triggers ONE full
    stateless resend and the session continues stateless afterwards."""
    provider = _FakeResponsesProvider("gpt-5.6-luna", fail_attempts={2})
    sess = OpenAISession(provider, role="episode:survey")

    b1 = [{"role": "user", "content": [{"type": "text", "text": "ctx"}]}]
    sess.send(b1)
    assert provider.calls[0]["store"] is True

    assistant = {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "decode_show", "input": {}}]}
    tool_result = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]}
    b2 = b1 + [assistant, tool_result]
    sess.send(b2)  # chained attempt (call 2) raises → stateless fallback

    # Call 2 was the failed chained attempt; call 3 is the full stateless resend.
    assert provider.calls[1]["store"] is True
    assert provider.calls[2]["store"] is False
    assert provider.calls[2]["previous_response_id"] is None
    assert provider.calls[2]["messages"] == b2

    # A later send continues STATELESS: no store=True, no previous_response_id,
    # full context transmitted.
    b3 = b2 + [{"role": "user", "content": [{"type": "text", "text": "more"}]}]
    sess.send(b3)
    assert provider.calls[3]["store"] is not True
    assert provider.calls[3]["previous_response_id"] is None
    assert provider.calls[3]["messages"] == b3


def test_openai_server_state_prefix_mismatch_falls_back_stateless():
    provider = _FakeResponsesProvider("gpt-5.6-luna")
    sess = OpenAISession(provider, role="episode:reading")
    b1 = [{"role": "user", "content": [{"type": "text", "text": "ctx"}]}]
    sess.send(b1)
    # A DIFFERENT block list (not a superset prefix) triggers a stateless resend.
    b2 = [{"role": "user", "content": [{"type": "text", "text": "totally different"}]}]
    sess.send(b2)
    assert provider.calls[1]["store"] is False
    assert provider.calls[1]["previous_response_id"] is None
    assert provider.calls[1]["messages"] == b2


def test_anthropic_session_applies_cache_control():
    provider = _FakeProvider("claude-sonnet-4-6", "anthropic")
    sess = AnthropicSession(provider)
    assert sess.capabilities.cache_breakpoints is True
    blocks = [{"role": "user", "content": [
        {"type": "text", "text": "stable", "cache_hint": True},
        {"type": "text", "text": "dynamic"},
    ]}]
    sess.send(blocks)
    sent = provider.calls[0]["messages"][0]["content"]
    # cache_hint became a cache_control breakpoint; the hint marker is gone.
    assert sent[0].get("cache_control") == {"type": "ephemeral"}
    assert "cache_hint" not in sent[0]
    assert "cache_control" not in sent[1]


def test_generic_session_strips_cache_hint():
    provider = _FakeProvider("qwen3:14b", "ollama")
    sess = make_lead_session(provider)
    blocks = [{"role": "user", "content": [
        {"type": "text", "text": "stable", "cache_hint": True}]}]
    sess.send(blocks)
    sent = provider.calls[0]["messages"][0]["content"]
    assert "cache_hint" not in sent[0]
    assert "cache_control" not in sent[0]


def test_r7_built_context_responses_converter_roundtrip():
    """R7: the merged [tool_results..., text] user turn the builder emits
    converts cleanly through the Responses input converter."""
    from agent.model_provider import _messages_to_openai_responses
    from investigation.context import build_lead_context
    from investigation.state import InvestigationState
    from agent.tools_v2 import NoGatesPolicy, WorkspaceToolExecutor
    from benchmark.loader import parse_canonical_transcription
    from workspace import Workspace

    ct = parse_canonical_transcription("S001 S002 | S003 S002")
    ws = Workspace(ct)
    state = InvestigationState(workspace=ws, language="en")
    state.recent_exchanges = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "decode_show",
             "input": {"branch": "main"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "{\"ok\": true}"}]},
    ]
    ex = WorkspaceToolExecutor(ws, "en", set(), [], {},
                               declaration_policy=NoGatesPolicy())
    messages = build_lead_context(state, ex, turn=2, token_budget=8000)
    items = _messages_to_openai_responses(messages)
    # The tool_result and the appended view text both survive conversion.
    assert any(it.get("type") == "function_call_output" and it.get("call_id") == "t1"
               for it in items)
    assert any(it.get("role") == "user" and it.get("content") for it in items)


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


def test_openai_chat_empty_assistant_message_not_null_content():
    """An assistant turn with no text and no tool_use (e.g. a one-shot verify
    worker that emitted nothing) must not serialize to content=null with no
    tool_calls — the OpenAI chat API rejects that on the nudge re-send. It is
    coerced to an empty string; well-formed messages are unaffected."""
    out = _messages_to_openai_chat([
        {"role": "user", "content": [{"type": "text", "text": "ctx"}]},
        {"role": "assistant", "content": []},                  # empty turn
        {"role": "user", "content": [{"type": "text", "text": "nudge"}]},
    ], system="sys")
    assistant = out[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "" and "tool_calls" not in assistant

    # Well-formed assistant text is unchanged.
    text_out = _messages_to_openai_chat(
        [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
    )
    assert text_out[0]["content"] == "hi"

    # A tool-only assistant keeps null content + tool_calls (valid for OpenAI).
    tool_out = _messages_to_openai_chat([
        {"role": "assistant",
         "content": [{"type": "tool_use", "id": "a", "name": "t", "input": {}}]}
    ])
    assert tool_out[0]["content"] is None and "tool_calls" in tool_out[0]
