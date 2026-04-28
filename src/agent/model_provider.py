"""Provider-neutral model interface for agent loops."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


class ModelProviderError(Exception):
    """Provider-neutral API error raised by model adapters."""


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
        try:
            response = self.api.send_message(
                messages=messages,
                tools=tools,
                system=system,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(str(exc)) from exc
        return normalize_model_response(response)


class OpenAIModelProvider:
    """Adapter for OpenAI chat-completions models with function tools."""

    provider_name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local extras
            raise ModelProviderError(
                "OpenAI provider requires the `openai` package. "
                "Install with: pip install -e '.[providers]'"
            ) from exc
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def send(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 4096,
    ) -> ModelResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=_messages_to_openai_chat(messages, system=system),
                tools=_tools_to_openai_chat(tools),
                max_completion_tokens=max_tokens,
            )
        except TypeError:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=_messages_to_openai_chat(messages, system=system),
                tools=_tools_to_openai_chat(tools),
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            if "max_completion_tokens" in str(exc):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=_messages_to_openai_chat(messages, system=system),
                        tools=_tools_to_openai_chat(tools),
                        max_tokens=max_tokens,
                    )
                except Exception as retry_exc:  # noqa: BLE001
                    raise ModelProviderError(str(retry_exc)) from retry_exc
            else:
                raise ModelProviderError(str(exc)) from exc
        return _openai_chat_response_to_model_response(response)


class GeminiModelProvider:
    """Adapter for Google Gemini models with function declarations."""

    provider_name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on local extras
            raise ModelProviderError(
                "Gemini provider requires the `google-genai` package. "
                "Install with: pip install -e '.[providers]'"
            ) from exc
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def send(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 4096,
    ) -> ModelResponse:
        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model,
                contents=_messages_to_gemini_contents(messages),
                config=types.GenerateContentConfig(
                    system_instruction=system or None,
                    max_output_tokens=max_tokens,
                    tools=_tools_to_gemini_tools(tools),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True,
                    ),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(str(exc)) from exc
        return _gemini_response_to_model_response(response)


def canonical_provider(provider: str | None) -> str:
    """Return the canonical provider key for CLI/config aliases."""

    value = (provider or "anthropic").strip().lower()
    aliases = {
        "claude": "anthropic",
        "anthropic": "anthropic",
        "openai": "openai",
        "gpt": "openai",
        "google": "gemini",
        "gemini": "gemini",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported provider: {provider}")
    return aliases[value]


def infer_provider_from_model(model: str | None, provider: str | None = None) -> str:
    """Infer provider from an explicit provider or a familiar model prefix."""

    if provider:
        return canonical_provider(provider)
    name = (model or "").strip().lower()
    if name.startswith("claude-"):
        return "anthropic"
    if name.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if name.startswith("gemini-"):
        return "gemini"
    return "anthropic"


def default_model_for_provider(provider: str) -> str:
    provider = canonical_provider(provider)
    if provider == "openai":
        return "gpt-5.4-mini"
    if provider == "gemini":
        return "gemini-3-flash-preview"
    return "claude-opus-4-7"


def make_model_provider(
    *,
    provider: str,
    api_key: str,
    model: str,
) -> AgentModelProvider:
    provider = canonical_provider(provider)
    if provider == "anthropic":
        from services.claude_api import ClaudeAPI

        return ClaudeModelProvider(ClaudeAPI(api_key=api_key, model=model))
    if provider == "openai":
        return OpenAIModelProvider(api_key=api_key, model=model)
    if provider == "gemini":
        return GeminiModelProvider(api_key=api_key, model=model)
    raise ValueError(f"Unsupported provider: {provider}")


def ensure_model_provider(api_or_provider: Any) -> AgentModelProvider:
    """Accept an existing provider or wrap the legacy ClaudeAPI object."""

    if (
        hasattr(api_or_provider, "provider_name")
        and hasattr(api_or_provider, "send")
    ):
        return api_or_provider
    return ClaudeModelProvider(api_or_provider)


_PRICING: dict[str, dict[str, tuple[float, float, float]]] = {
    "anthropic": {
        "claude-opus-4": (15.00, 75.00, 1.50),
        "claude-sonnet-4": (3.00, 15.00, 0.30),
        "claude-haiku-4": (0.80, 4.00, 0.08),
        "claude-opus-3": (15.00, 75.00, 1.50),
        "claude-sonnet-3": (3.00, 15.00, 0.30),
        "claude-haiku-3": (0.25, 1.25, 0.03),
    },
    "openai": {
        "gpt-5.4-mini": (0.80, 2.00, 0.08),
        "gpt-5.4": (2.00, 8.00, 0.20),
        "gpt-5": (1.25, 10.00, 0.125),
    },
    "gemini": {
        "gemini-3-flash-lite": (0.25, 1.00, 0.025),
        "gemini-3-flash-preview": (0.50, 2.00, 0.05),
        "gemini-3-flash": (0.50, 2.00, 0.05),
        "gemini-3.1-flash-lite-preview": (0.25, 1.00, 0.025),
        "gemini-3.1-flash-lite": (0.25, 1.00, 0.025),
        "gemini-3.1-flash": (0.50, 2.00, 0.05),
        "gemini-3.1-pro": (2.00, 12.00, 0.20),
    },
}


def estimate_provider_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Return approximate USD cost for normalized usage counters."""

    provider = canonical_provider(provider)
    pricing = _PRICING.get(provider, {})
    prefix = ""
    for candidate in sorted(pricing, key=len, reverse=True):
        if model.startswith(candidate):
            prefix = candidate
            break
    if not prefix:
        return 0.0
    inp_rate, out_rate, cache_rate = pricing[prefix]
    billed_input = max(0, input_tokens - cache_read_tokens)
    return (
        billed_input * inp_rate / 1_000_000
        + cache_read_tokens * cache_rate / 1_000_000
        + output_tokens * out_rate / 1_000_000
    )


def _tools_to_openai_chat(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    out = []
    for tool in tools:
        out.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return out


def _messages_to_openai_chat(
    messages: list[dict[str, Any]],
    *,
    system: str = "",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": str(block.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name", "")),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    })
            chat_message: dict[str, Any] = {
                "role": "assistant",
                "content": "\n\n".join(t for t in text_parts if t) or None,
            }
            if tool_calls:
                chat_message["tool_calls"] = tool_calls
            out.append(chat_message)
        elif role == "user" and isinstance(content, list):
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    out.append({
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id", "")),
                        "content": str(block.get("content", "")),
                    })
                elif block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
            if text_parts:
                out.append({"role": "user", "content": "\n\n".join(text_parts)})
        else:
            out.append({"role": role, "content": _content_to_text(content)})
    return out


def _openai_chat_response_to_model_response(response: Any) -> ModelResponse:
    choice = response.choices[0]
    message = choice.message
    content: list[ModelContentBlock] = []
    if getattr(message, "content", None):
        content.append(TextBlock(text=str(message.content)))
    for tool_call in getattr(message, "tool_calls", None) or []:
        raw_args = getattr(tool_call.function, "arguments", "{}") or "{}"
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed = {}
        content.append(
            ToolUseBlock(
                id=str(tool_call.id),
                name=str(tool_call.function.name),
                input=parsed if isinstance(parsed, dict) else {},
            )
        )
    raw_usage = getattr(response, "usage", None)
    prompt_details = getattr(raw_usage, "prompt_tokens_details", None)
    usage = ModelUsage(
        input_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(prompt_details, "cached_tokens", 0) or 0),
    )
    return ModelResponse(content=content, usage=usage, raw=response)


def _tools_to_gemini_tools(tools: list[dict[str, Any]] | None) -> list[Any] | None:
    if not tools:
        return None
    try:
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise ModelProviderError(
            "Gemini provider requires the `google-genai` package. "
            "Install with: pip install -e '.[providers]'"
        ) from exc
    declarations = []
    for tool in tools:
        declarations.append(
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=_schema_for_gemini(tool.get("input_schema", {})),
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _messages_to_gemini_contents(messages: list[dict[str, Any]]) -> list[Any]:
    try:
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise ModelProviderError(
            "Gemini provider requires the `google-genai` package. "
            "Install with: pip install -e '.[providers]'"
        ) from exc

    out: list[Any] = []
    for message in messages:
        role = "model" if message.get("role") == "assistant" else "user"
        content = message.get("content", "")
        parts: list[Any] = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(types.Part.from_text(text=str(block.get("text", ""))))
                elif block_type == "tool_use":
                    name = str(block.get("name", ""))
                    parts.append(types.Part.from_text(
                        text=(
                            "[assistant requested tool] "
                            f"{name}({json.dumps(block.get('input') or {}, sort_keys=True)})"
                        ),
                    ))
                elif block_type == "tool_result":
                    tool_id = str(block.get("tool_use_id", ""))
                    parts.append(types.Part.from_text(
                        text=(
                            "[tool result] "
                            f"{tool_id}: {str(block.get('content', ''))}"
                        ),
                    ))
        else:
            parts.append(types.Part.from_text(text=_content_to_text(content)))
        if parts:
            out.append(types.Content(role=role, parts=parts))
    return out


def _gemini_response_to_model_response(response: Any) -> ModelResponse:
    content: list[ModelContentBlock] = []
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                content.append(TextBlock(text=str(text)))
            function_call = getattr(part, "function_call", None)
            if function_call:
                args = getattr(function_call, "args", {}) or {}
                try:
                    parsed_args = dict(args)
                except Exception:  # noqa: BLE001
                    parsed_args = {}
                content.append(
                    ToolUseBlock(
                        id=f"gemini_{uuid.uuid4().hex[:12]}",
                        name=str(getattr(function_call, "name", "")),
                        input=parsed_args,
                    )
                )
    elif getattr(response, "text", None):
        content.append(TextBlock(text=str(response.text)))
    raw_usage = getattr(response, "usage_metadata", None)
    usage = ModelUsage(
        input_tokens=int(getattr(raw_usage, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(raw_usage, "candidates_token_count", 0) or 0),
        cache_read_input_tokens=int(getattr(raw_usage, "cached_content_token_count", 0) or 0),
    )
    return ModelResponse(content=content, usage=usage, raw=response)


def _schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Trim JSON Schema features that Gemini function declarations reject."""

    allowed = {
        "type",
        "properties",
        "required",
        "enum",
        "items",
        "description",
        "nullable",
    }
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in allowed:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {
                str(prop): _schema_for_gemini(prop_schema)
                for prop, prop_schema in value.items()
                if isinstance(prop_schema, dict)
            }
        elif key == "items" and isinstance(value, dict):
            out[key] = _schema_for_gemini(value)
        elif key == "enum" and isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                out[key] = value
        else:
            out[key] = value
    if not out:
        return {"type": "object", "properties": {}}
    return out


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n\n".join(parts)
    return str(content)
