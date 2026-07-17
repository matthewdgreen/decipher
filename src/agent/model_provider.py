"""Provider-neutral model interface for agent loops."""
from __future__ import annotations

import dataclasses
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


class ModelProviderError(Exception):
    """Provider-neutral API error raised by model adapters."""


# --- Transient rate-limit retry (2026-07-17, K3/OpenRouter incident) --------
# An upstream 429 ("temporarily rate-limited upstream ... Retry-After: 1")
# killed a whole v3 run on turn 2. Transient limits deserve a few short,
# bounded retries; QUOTA exhaustion (insufficient_quota, also a 429) does not —
# retrying an unfunded account is futile (observed live on the Sequence-C
# smoke) and must fail fast to the caller's honest error path.

_RATE_LIMIT_MARKERS = ("rate-limit", "rate limit", "rate_limit")
_NON_RETRYABLE_MARKERS = ("insufficient_quota",)
_RATE_LIMIT_RETRY_DELAYS = (2.0, 5.0, 10.0)


def is_retryable_rate_limit_error(exc: BaseException) -> bool:
    """True for a transient 429/rate-limit error; False for quota exhaustion
    or anything else."""
    text = str(exc).lower()
    if any(marker in text for marker in _NON_RETRYABLE_MARKERS):
        return False
    if re.search(r"\b429\b", text):
        return True
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def parse_retry_after_seconds(exc: BaseException) -> float | None:
    """Best-effort Retry-After (seconds) from a provider error string.

    Matches both header form (``'Retry-After': '1'``) and OpenRouter metadata
    (``'retry_after_seconds': 1``). Clamped to [0, 60]; None when absent.
    """
    match = re.search(
        r"retry[_-]after[^0-9]{0,24}?(\d+(?:\.\d+)?)", str(exc), re.IGNORECASE
    )
    if not match:
        return None
    try:
        return min(max(float(match.group(1)), 0.0), 60.0)
    except ValueError:
        return None


def call_with_rate_limit_retry(
    send: Callable[[], Any],
    *,
    delays: tuple[float, ...] = _RATE_LIMIT_RETRY_DELAYS,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> Any:
    """Run ``send()``, retrying transient rate-limit errors with short waits.

    At most ``len(delays)`` retries; each wait is the scheduled delay or the
    provider's parsed Retry-After, whichever is longer. A persistent limit
    re-raises on the final attempt so the caller's normal error path stays
    the terminal. Non-rate-limit errors (including insufficient_quota)
    propagate immediately. KeyboardInterrupt is never caught.
    """
    for attempt, delay in enumerate(delays, start=1):
        try:
            return send()
        except ModelProviderError as exc:
            if not is_retryable_rate_limit_error(exc):
                raise
            wait = max(delay, parse_retry_after_seconds(exc) or 0.0)
            if on_retry is not None:
                on_retry(attempt, wait, exc)
            time.sleep(wait)
    return send()


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


@dataclass(frozen=True)
class ProviderExtraBlock:
    """An opaque provider-specific passthrough block.

    Used to carry provider items that the harness does not interpret but must
    round-trip verbatim between turns — currently OpenAI ``reasoning`` output
    items (with ``encrypted_content``) for reasoning-model tool-call passback.
    It is serialized into the assistant turn's message history as a plain dict
    ``{"type": "provider_extra", "provider": ..., "kind": ..., "items": [...]}``
    and re-emitted to its native shape only by the originating provider's
    converter; every other provider's converter drops it.
    """

    provider: str
    kind: str
    items: list[dict[str, Any]] = field(default_factory=list)
    type: str = "provider_extra"


ModelContentBlock = TextBlock | ToolUseBlock | ProviderExtraBlock


def _reasoning_passback_enabled() -> bool:
    """Whether OpenAI reasoning-item capture/re-emit is active.

    Default on; ``DECIPHER_OPENAI_REASONING_PASSBACK=0`` (or ``false``/``no``/
    ``off``) disables it for A/B measurement.
    """
    import os

    raw = os.environ.get("DECIPHER_OPENAI_REASONING_PASSBACK", "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ModelResponse:
    """A normalized assistant response used by Decipher's agent harness."""

    content: list[ModelContentBlock]
    usage: ModelUsage = field(default_factory=ModelUsage)
    raw: Any = None


def served_model_from_response(response: Any) -> str | None:
    """Return the model id the provider actually served, if it exposes one.

    All three primary providers (Anthropic Message, OpenAI ChatCompletion /
    Responses objects) expose ``.model`` on their raw response. ``ModelResponse``
    round-trips the raw object via ``.raw``. Returns ``None`` when unavailable
    (e.g. a provider that does not surface it) so callers treat "unknown" as
    "not a gate hit" rather than a false positive.
    """
    if response is None:
        return None
    raw = response.raw if isinstance(response, ModelResponse) else response
    served = getattr(raw, "model", None)
    if served is None and isinstance(raw, dict):
        served = raw.get("model")
    if served is None:
        return None
    served = str(served).strip()
    return served or None


def served_model_matches(requested: str, served: str | None) -> bool:
    """Whether a served model id matches the requested one (no safety gate).

    Providers routinely append a date/build suffix to the served id
    (``gpt-5.5`` -> ``gpt-5.5-2026-01-01``; ``claude-fable-5`` ->
    ``claude-fable-5-20260101``), so a prefix relationship in either direction
    counts as a match. A missing served id is treated as a match (unknown, not a
    gate hit). A genuinely different family (``claude-fable-5`` served as
    ``claude-opus-4-8``) does not match -> the gate fired.
    """
    if not served:
        return True
    req = (requested or "").strip().lower()
    srv = served.strip().lower()
    if not req:
        return True
    return req == srv or srv.startswith(req) or req.startswith(srv)


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


def _opaque_block_to_dict(block: Any) -> dict[str, Any]:
    """Serialize an unrecognized response block to a plain history dict."""
    if isinstance(block, dict):
        return dict(block)
    if dataclasses.is_dataclass(block):
        return dataclasses.asdict(block)
    return {"type": getattr(block, "type", "unknown")}


def _collect_assistant_blocks(
    response: ModelResponse,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Split a ModelResponse into history blocks, tool-use calls, and text.

    Returns ``(assistant_blocks, tool_uses, text_parts)``:
      * ``assistant_blocks`` is the assistant turn's ``content`` list, ready to
        append to the message history (text, tool_use, and opaque
        provider_extra blocks preserved verbatim).
      * ``tool_uses`` is the subset of tool_use blocks the loop must execute.
      * ``text_parts`` collects the assistant's text for logging/plan capture.

    Hoisted from ``loop_v2`` (F1/F2) so the v3 session layer does not import
    from the v2 loop. Behavior-preserving.
    """
    assistant_blocks: list[dict[str, Any]] = []
    tool_uses: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for block in response.content:
        block_type = (
            block.get("type") if isinstance(block, dict)
            else getattr(block, "type", None)
        )
        if block_type == "text":
            text = block.get("text", "") if isinstance(block, dict) else block.text
            assistant_blocks.append({"type": "text", "text": text})
            text_parts.append(text)
        elif block_type == "tool_use":
            if isinstance(block, dict):
                b_id, b_name, b_input = (
                    block.get("id"),
                    block.get("name"),
                    block.get("input") or {},
                )
            else:
                b_id, b_name, b_input = block.id, block.name, block.input
            assistant_blocks.append({
                "type": "tool_use",
                "id": b_id,
                "name": b_name,
                "input": b_input,
            })
            tool_uses.append({
                "id": b_id,
                "name": b_name,
                "input": b_input,
            })
        else:
            # Opaque / unknown block (e.g. a provider_extra reasoning-passback
            # block): preserve it verbatim in the assistant turn's history
            # without treating it as text or a tool call.
            assistant_blocks.append(_opaque_block_to_dict(block))
    return assistant_blocks, tool_uses, text_parts


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
                messages=_strip_provider_extra_blocks(messages),
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
        previous_response_id: str | None = None,
        store: bool | None = None,
    ) -> ModelResponse:
        # GPT-5.6 tiers reject function tools on /v1/chat/completions unless
        # reasoning_effort is 'none' (which would lobotomize the model).  Route
        # them through /v1/responses instead.  gpt-5.5 and earlier keep the
        # chat-completions path below unchanged.
        #
        # ``previous_response_id`` / ``store`` support within-episode server-state
        # chaining (C7): they are only ever set by an episode-role OpenAISession
        # on the Responses path.  For every other caller they default to None and
        # the request is byte-identical to the pre-M2 behavior.
        if _requires_responses_api(self.model):
            return self._send_responses(
                messages=messages,
                tools=tools,
                system=system,
                max_tokens=max_tokens,
                previous_response_id=previous_response_id,
                store=store,
            )
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

    def _send_responses(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        system: str,
        max_tokens: int,
        previous_response_id: str | None = None,
        store: bool | None = None,
    ) -> ModelResponse:
        """Send one request through the OpenAI Responses API.

        Used for reasoning tiers (e.g. gpt-5.6-*) that reject function tools on
        /v1/chat/completions.  No ``reasoning`` parameter is set, so the model's
        server-side default effort applies.

        ``store``/``previous_response_id`` control within-episode server-state
        chaining (C7/F2).  When ``store is None`` (every non-episode caller) the
        request is byte-identical to the pre-M2 behavior.  When ``store`` is set
        explicitly, ``include=["reasoning.encrypted_content"]`` is always kept —
        it enables the stateless fallback and keeps the transcript
        self-contained — and ``previous_response_id`` is attached only for a
        chained (``store=True``) send.
        """
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": _messages_to_openai_responses(messages),
            "max_output_tokens": max_tokens,
        }
        # Omit the tools kwarg entirely for a tool-less send (e.g. an episode's
        # final "emit the result JSON" nudge): `"tools": null` may 400 on the
        # Responses API. Every pre-M2 live caller passed a non-empty tools list,
        # so this changes nothing for existing paths (kwargs-pinned).
        converted_tools = _tools_to_openai_responses(tools)
        if converted_tools is not None:
            create_kwargs["tools"] = converted_tools
        if store is None:
            # Default (M1) behavior — byte-identical to pre-M2.
            if _reasoning_passback_enabled():
                # Ask for reasoning items with re-sendable encrypted payloads so
                # they can be passed back between tool calls (OpenAI reasoning-
                # model recommendation).  store=False keeps this adapter
                # stateless — the full input, including prior reasoning items, is
                # re-sent each turn.
                create_kwargs["store"] = False
                create_kwargs["include"] = ["reasoning.encrypted_content"]
        else:
            create_kwargs["store"] = store
            create_kwargs["include"] = ["reasoning.encrypted_content"]
            if store and previous_response_id is not None:
                create_kwargs["previous_response_id"] = previous_response_id
        try:
            response = self.client.responses.create(**create_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(str(exc)) from exc
        return _openai_responses_response_to_model_response(response)


class OllamaModelProvider:
    """Adapter for locally-running Ollama models.

    Ollama exposes an OpenAI-compatible REST API at http://localhost:11434/v1,
    so this adapter reuses the OpenAI chat-completions path with a custom
    base URL.  No API key is required.

    Environment variables
    ---------------------
    OLLAMA_HOST     Override base URL (default: http://localhost:11434).
    OLLAMA_NUM_CTX  Context window size passed to Ollama as num_ctx
                    (default: 32768).  Decipher prompts are typically
                    20 K+ tokens; Ollama's built-in model default (often
                    4 096) will silently truncate them.
    OLLAMA_TIMEOUT  Per-request timeout in seconds (default: 120).
                    Ollama returns HTTP 500 after its own internal timeout
                    (~10 min); setting a client-side limit avoids waiting
                    that long before surfacing the failure.
    """

    provider_name = "ollama"
    # Minimum context window to request from Ollama.  Decipher's initial
    # user message for a typical benchmark case is 20-25 K tokens; the
    # Ollama default of 4 096 silently truncates to almost nothing.
    DEFAULT_NUM_CTX = 32768
    DEFAULT_TIMEOUT = 120.0

    def __init__(self, model: str) -> None:
        import os

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ModelProviderError(
                "Ollama provider requires the `openai` package. "
                "Install with: pip install -e '.[providers]'"
            ) from exc
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        try:
            self.num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", self.DEFAULT_NUM_CTX))
        except ValueError:
            self.num_ctx = self.DEFAULT_NUM_CTX
        try:
            timeout = float(os.environ.get("OLLAMA_TIMEOUT", self.DEFAULT_TIMEOUT))
        except ValueError:
            timeout = self.DEFAULT_TIMEOUT
        self.model = model
        self.client = OpenAI(
            base_url=f"{host.rstrip('/')}/v1",
            api_key="ollama",
            timeout=timeout,
        )

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
                max_tokens=max_tokens,
                extra_body={"options": {"num_ctx": self.num_ctx}},
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(str(exc)) from exc
        return _openai_chat_response_to_model_response(response)


class OpenRouterModelProvider:
    """Adapter for OpenRouter's OpenAI-compatible gateway.

    OpenRouter proxies 200+ open-weight and frontier models through a single
    OpenAI-compatible endpoint.  Model IDs use the 'provider/name' format,
    e.g. 'meta-llama/llama-3.3-70b-instruct' or 'deepseek/deepseek-v4-pro'.

    API key
    -------
    Set OPENROUTER_API_KEY, put it in .env, put it in
    .decipher_keys/openrouter_api_key, or store it in the macOS Keychain
    under service='decipher', account='openrouter_api_key'.

    Environment variables
    ---------------------
    OPENROUTER_TIMEOUT   Per-request timeout in seconds (default: 180).
                         Open-weight models behind OpenRouter can be slower
                         than frontier APIs; 180 s is a conservative safe
                         default for long tool-heavy agent turns.
    OPENROUTER_SITE_URL  HTTP-Referer header sent with every request.
                         Appears in your OpenRouter dashboard and helps with
                         rate-limit allowances.  Defaults to the project URL.
    OPENROUTER_APP_NAME  X-Title header for dashboard identification.
    """

    provider_name = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_TIMEOUT = 180.0

    def __init__(self, api_key: str, model: str) -> None:
        import os

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ModelProviderError(
                "OpenRouter provider requires the `openai` package. "
                "Install with: pip install -e '.[providers]'"
            ) from exc
        try:
            timeout = float(os.environ.get("OPENROUTER_TIMEOUT", self.DEFAULT_TIMEOUT))
        except ValueError:
            timeout = self.DEFAULT_TIMEOUT
        site_url = os.environ.get(
            "OPENROUTER_SITE_URL",
            "https://github.com/decipher-research/decipher",
        )
        app_name = os.environ.get("OPENROUTER_APP_NAME", "decipher")
        self.model = model
        self.client = OpenAI(
            base_url=self.BASE_URL,
            api_key=api_key,
            timeout=timeout,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
        )

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
                messages=_messages_to_openrouter_chat(messages, system=system),
                tools=_tools_to_openai_chat(tools),
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(str(exc)) from exc
        return _openrouter_chat_response_to_model_response(response)


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
        "ollama": "ollama",
        "openrouter": "openrouter",
        "or": "openrouter",
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
    # OpenRouter model IDs use "provider/model-name" format.
    if "/" in name:
        return "openrouter"
    return "anthropic"


def default_model_for_provider(provider: str) -> str:
    provider = canonical_provider(provider)
    if provider == "openai":
        return "gpt-5.5"
    if provider == "gemini":
        return "gemini-3-flash-preview"
    if provider == "ollama":
        return "qwen3:14b"
    if provider == "openrouter":
        return "meta-llama/llama-3.3-70b-instruct"
    return "claude-sonnet-4-6"


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
    if provider == "ollama":
        return OllamaModelProvider(model=model)
    if provider == "openrouter":
        return OpenRouterModelProvider(api_key=api_key, model=model)
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
    # Ollama models run locally — no per-token cost.
    # Models listed here are the ones with documented tool-calling support;
    # any model name accepted by the local Ollama instance will also work.
    "ollama": {
        "qwen3:8b": (0.0, 0.0, 0.0),
        "qwen3:14b": (0.0, 0.0, 0.0),
        "qwen3:30b-a3b": (0.0, 0.0, 0.0),
        "qwen3:32b": (0.0, 0.0, 0.0),
        "llama3.1:8b": (0.0, 0.0, 0.0),
        "llama3.1:70b": (0.0, 0.0, 0.0),
        "mistral-nemo": (0.0, 0.0, 0.0),
    },
    "anthropic": {
        "claude-opus-4": (15.00, 75.00, 1.50),
        "claude-sonnet-4": (3.00, 15.00, 0.30),
        "claude-haiku-4": (0.80, 4.00, 0.08),
        "claude-opus-3": (15.00, 75.00, 1.50),
        "claude-sonnet-3": (3.00, 15.00, 0.30),
        "claude-haiku-3": (0.25, 1.25, 0.03),
    },
    "openai": {
        # gpt-5.5/5.6 rates supplied by Matthew from the OpenAI pricing page
        # (2026-07-13). Cache-write premiums are not modeled by
        # estimate_provider_cost, so estimates slightly undercount runs with
        # heavy cache churn.
        # Pro tiers have no cached-input rate on the pricing page; cached is
        # set equal to input (no discount) so estimates stay conservative.
        "gpt-5.6-sol": (5.00, 30.00, 0.50),
        "gpt-5.6-terra": (2.50, 15.00, 0.25),
        "gpt-5.6-luna": (1.00, 6.00, 0.10),
        "gpt-5.5-pro": (30.00, 180.00, 30.00),
        "gpt-5.5": (5.00, 30.00, 0.50),
        "gpt-5.4-pro": (30.00, 180.00, 30.00),
        "gpt-5.4-mini": (0.75, 4.50, 0.075),
        "gpt-5.4-nano": (0.20, 1.25, 0.02),
        "gpt-5.4": (2.50, 15.00, 0.25),
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
    # OpenRouter: prices are approximate (USD/M tokens) and may vary by route.
    # Prefix-matching is longest-first, so more specific entries win.
    "openrouter": {
        # Meta Llama 3.x / 4.x
        "meta-llama/llama-4-maverick": (0.20, 0.80, 0.020),
        "meta-llama/llama-4-scout": (0.10, 0.35, 0.010),
        "meta-llama/llama-3.3-70b-instruct": (0.28, 0.56, 0.028),
        "meta-llama/llama-3.1-8b-instruct": (0.04, 0.04, 0.004),
        # DeepSeek
        "deepseek/deepseek-r1-0528": (0.55, 2.19, 0.055),
        "deepseek/deepseek-r1": (0.55, 2.19, 0.055),
        "deepseek/deepseek-v3-0324": (0.28, 0.89, 0.028),
        "deepseek/deepseek-v3": (0.28, 0.89, 0.028),
        # Mistral / Mixtral
        "mistralai/mistral-small-3.2-24b-instruct": (0.10, 0.30, 0.010),
        "mistralai/mistral-small-3.1-24b-instruct": (0.10, 0.30, 0.010),
        "mistralai/mistral-nemo": (0.065, 0.065, 0.007),
        # Qwen (Alibaba)
        "qwen/qwen3-30b-a3b": (0.10, 0.30, 0.010),
        "qwen/qwen3-14b": (0.10, 0.30, 0.010),
        "qwen/qwen3-8b": (0.04, 0.04, 0.004),
        # Google via OpenRouter
        "google/gemini-2.5-flash-preview": (0.15, 0.60, 0.015),
        "google/gemini-2.0-flash-001": (0.10, 0.40, 0.010),
        # xAI Grok
        "x-ai/grok-3-mini-beta": (0.30, 0.50, 0.030),
        # Nous Hermes
        "nousresearch/hermes-3-llama-3.1-70b": (0.40, 0.40, 0.040),
    },
}


# ---------------------------------------------------------------------------
# OpenRouter live pricing
# ---------------------------------------------------------------------------

# In-memory cache: populated once per process from disk or network.
_OPENROUTER_PRICING_LIVE: dict[str, tuple[float, float, float]] | None = None
# Full set of all known OpenRouter model IDs (including free/unpriced ones).
# Populated alongside _OPENROUTER_PRICING_LIVE whenever we touch the models API.
_OPENROUTER_ALL_MODEL_IDS: set[str] | None = None
# Guard so we attempt the network fetch at most once per process.
_OPENROUTER_FETCH_ATTEMPTED: bool = False
# Disk cache is considered stale after this many hours.
_OPENROUTER_CACHE_TTL_HOURS: int = 24


def _default_openrouter_cache_path() -> Path:
    return Path.home() / ".config" / "decipher" / "openrouter_pricing.json"


def _parse_openrouter_models_response(
    data: dict[str, Any],
) -> dict[str, tuple[float, float, float]]:
    """Convert a raw /api/v1/models response into our ($/M-in, $/M-out, $/M-cache) format."""
    result: dict[str, tuple[float, float, float]] = {}
    for model in data.get("data", []) or []:
        model_id = str(model.get("id") or "").strip()
        if not model_id:
            continue
        pricing = model.get("pricing") or {}
        try:
            # OpenRouter exposes per-token prices as decimal strings.
            input_rate = float(pricing.get("prompt") or 0) * 1_000_000
            output_rate = float(pricing.get("completion") or 0) * 1_000_000
        except (ValueError, TypeError):
            continue
        if input_rate <= 0.0 and output_rate <= 0.0:
            continue  # skip free, unpriced, and routing-utility models
        # OpenRouter doesn't expose a separate cache-read rate; approximate at
        # 10 % of input (consistent with provider cache-read pricing norms).
        cache_rate = round(input_rate * 0.10, 6)
        result[model_id] = (round(input_rate, 6), round(output_rate, 6), cache_rate)
    return result


def _load_openrouter_disk_cache(
    cache_path: Path | None = None,
) -> dict[str, tuple[float, float, float]] | None:
    """Return disk-cached pricing if the file exists and is within TTL.

    As a side-effect, populates ``_OPENROUTER_ALL_MODEL_IDS`` from the
    ``all_model_ids`` field written by recent cache versions.
    """
    global _OPENROUTER_ALL_MODEL_IDS
    import time

    path = cache_path or _default_openrouter_cache_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        age_hours = (time.time() - float(raw.get("fetched_at", 0))) / 3600
        if age_hours > _OPENROUTER_CACHE_TTL_HOURS:
            return None
        pricing: dict[str, tuple[float, float, float]] = {}
        for model_id, rates in (raw.get("models") or {}).items():
            if isinstance(rates, (list, tuple)) and len(rates) >= 3:
                pricing[model_id] = (float(rates[0]), float(rates[1]), float(rates[2]))
        # Populate all-IDs set from cache if available (written by recent versions).
        if _OPENROUTER_ALL_MODEL_IDS is None and raw.get("all_model_ids"):
            _OPENROUTER_ALL_MODEL_IDS = set(raw["all_model_ids"])
        return pricing or None
    except Exception:  # noqa: BLE001
        return None


def fetch_openrouter_pricing(
    *,
    timeout: float = 10.0,
    cache_path: Path | None = None,
    write_cache: bool = True,
) -> dict[str, tuple[float, float, float]]:
    """Fetch current OpenRouter model pricing from their public models API.

    No authentication is required.  Writes a timestamped JSON disk cache so
    subsequent process starts don't need a network round-trip.

    Returns
    -------
    dict mapping model_id → (input_$/M, output_$/M, cache_read_$/M).

    Raises
    ------
    OSError / urllib.error.URLError on network failure.
    ValueError on unexpected response format.
    """
    import time
    import urllib.request

    url = "https://openrouter.ai/api/v1/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        data = json.loads(resp.read())

    pricing = _parse_openrouter_models_response(data)
    if not pricing:
        raise ValueError("OpenRouter /api/v1/models returned no priced models")

    # Collect all model IDs (including free/unpriced) for validation use.
    all_ids: set[str] = {
        str(m.get("id", "")).strip()
        for m in (data.get("data") or [])
        if m.get("id")
    }
    global _OPENROUTER_ALL_MODEL_IDS
    _OPENROUTER_ALL_MODEL_IDS = all_ids

    if write_cache:
        path = cache_path or _default_openrouter_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        cache_data: dict[str, Any] = {
            "fetched_at": time.time(),
            "model_count": len(pricing),
            "models": {k: list(v) for k, v in pricing.items()},
            # Store all IDs (not just priced) so validation can use the cache.
            "all_model_ids": sorted(all_ids),
        }
        path.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")

    return pricing


def _get_openrouter_live_pricing(
    cache_path: Path | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Return live OpenRouter pricing, using the fastest available source.

    Priority order:
    1. In-memory cache (populated by a prior call this process).
    2. Disk cache (if the file is < ``_OPENROUTER_CACHE_TTL_HOURS`` hours old).
    3. Network fetch (attempted at most once per process; silently skipped on
       failure so cost estimation degrades gracefully to the hardcoded table).

    Returns an empty dict if no live data is available.
    """
    global _OPENROUTER_PRICING_LIVE, _OPENROUTER_FETCH_ATTEMPTED, _OPENROUTER_ALL_MODEL_IDS

    if _OPENROUTER_PRICING_LIVE is not None:
        return _OPENROUTER_PRICING_LIVE

    disk = _load_openrouter_disk_cache(cache_path)
    if disk:
        _OPENROUTER_PRICING_LIVE = disk
        return disk

    if not _OPENROUTER_FETCH_ATTEMPTED:
        _OPENROUTER_FETCH_ATTEMPTED = True
        try:
            pricing = fetch_openrouter_pricing(
                timeout=5.0,  # short timeout — don't slow runs if OR is down
                cache_path=cache_path,
                write_cache=True,
            )
            _OPENROUTER_PRICING_LIVE = pricing
            return pricing
        except Exception:  # noqa: BLE001
            pass  # fall through to hardcoded table

    return {}


def validate_model(
    provider: str,
    model: str,
    *,
    cache_path: Path | None = None,
    timeout: float = 5.0,
) -> tuple[bool, str]:
    """Check whether a model is likely valid for the given provider.

    Returns ``(True, "")`` when the model appears valid or cannot be verified
    (e.g. the network is unreachable).  Returns ``(False, human_readable_hint)``
    when the model is definitively not recognised.

    For OpenRouter the check is authoritative: we fetch (or use the cached)
    ``/api/v1/models`` list which covers all 300+ models.  For other providers
    the check is best-effort only (prefix heuristics; never blocks on failure).
    """
    provider = canonical_provider(provider)

    if provider == "openrouter":
        # Ensure the all-IDs set is populated.  _get_openrouter_live_pricing()
        # triggers the disk-cache load and lazy network fetch as a side-effect.
        _get_openrouter_live_pricing(cache_path=cache_path)
        # Tracks whether the id set below reflects a live fetch made THIS call
        # (vs. a possibly-stale disk cache). Gates the single self-heal refresh.
        fetched_fresh = False

        if _OPENROUTER_ALL_MODEL_IDS is None:
            # Disk cache exists but predates the all_model_ids field (old format),
            # or no cache exists yet.  Do a direct fetch to populate IDs; this
            # also rewrites the cache in the new format.
            try:
                fetch_openrouter_pricing(timeout=timeout, cache_path=cache_path)
                fetched_fresh = True
            except Exception:  # noqa: BLE001
                return True, ""  # Cannot reach OpenRouter — don't block the run.

        if _OPENROUTER_ALL_MODEL_IDS is not None:
            if model in _OPENROUTER_ALL_MODEL_IDS:
                return True, ""
            # Self-heal a stale cache before rejecting: the id set may have come
            # from a disk cache (<=24h old) written BEFORE a freshly-listed model
            # existed on OpenRouter. When it did NOT come from a fetch we just
            # ran (``fetched_fresh``), force exactly ONE live refresh and
            # re-check. A genuinely invalid id still falls through to the hint
            # below — this is bounded to a single extra fetch, never a loop.
            if not fetched_fresh:
                try:
                    fetch_openrouter_pricing(timeout=timeout, cache_path=cache_path)
                except Exception:  # noqa: BLE001
                    pass  # network unreachable — fall through to the not-found hint
                else:
                    if _OPENROUTER_ALL_MODEL_IDS and model in _OPENROUTER_ALL_MODEL_IDS:
                        return True, ""
            # Build a helpful suggestion list: models whose ID contains any
            # component of the supplied name (split on "/"). Uses the freshest
            # id set available after the refresh above.
            parts = [p.lower() for p in model.replace(":", "/").split("/") if p]
            suggestions = sorted(
                mid for mid in (_OPENROUTER_ALL_MODEL_IDS or set())
                if any(p in mid.lower() for p in parts)
            )[:5]
            hint = f"Model '{model}' was not found on OpenRouter."
            if suggestions:
                hint += "\nDid you mean one of these?\n" + "\n".join(
                    f"  {s}" for s in suggestions
                )
            hint += (
                "\nRun 'decipher doctor' to see known models, "
                "or browse https://openrouter.ai/models"
            )
            return False, hint

        # Could not reach OpenRouter at all — don't block.
        return True, ""

    # For other providers: lightweight prefix check.  We never hard-fail here
    # because our known-model lists are incomplete.
    if provider == "ollama":
        return True, ""  # Ollama validates at connection time; no API for lookup.

    known_prefixes = list(_PRICING.get(provider, {}).keys())
    if known_prefixes and not any(model.startswith(p) for p in known_prefixes):
        # Model name doesn't match any known prefix for this provider — warn
        # but still allow (user may be using a newer model not in our table).
        return True, (
            f"Warning: '{model}' is not in the known model list for {provider}. "
            "The run will proceed; if the model name is wrong the API will error."
        )
    return True, ""


def estimate_provider_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Return approximate USD cost for normalized usage counters."""

    provider = canonical_provider(provider)
    if provider == "openrouter":
        # Merge: live data overrides the hardcoded table when available; the
        # hardcoded table fills gaps for models the live fetch doesn't cover.
        pricing: dict[str, tuple[float, float, float]] = {
            **_PRICING.get("openrouter", {}),
            **_get_openrouter_live_pricing(),
        }
    else:
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
    passthrough_provider: str | None = None,
    drop_empty_assistant: bool = False,
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
            provider_reasoning: dict[str, Any] = {}
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
                elif (
                    passthrough_provider
                    and block.get("type") == "provider_extra"
                    and block.get("provider") == passthrough_provider
                    and block.get("kind") == "reasoning"
                ):
                    for item in block.get("items") or []:
                        if not isinstance(item, dict):
                            continue
                        details = item.get("reasoning_details")
                        if isinstance(details, list):
                            provider_reasoning["reasoning_details"] = details
                        reasoning = item.get("reasoning")
                        if isinstance(reasoning, str) and reasoning:
                            provider_reasoning["reasoning"] = reasoning
            chat_message: dict[str, Any] = {
                "role": "assistant",
                "content": "\n\n".join(t for t in text_parts if t) or None,
            }
            if tool_calls:
                chat_message["tool_calls"] = tool_calls
            elif chat_message["content"] is None:
                if drop_empty_assistant:
                    # Some OpenAI-compatible providers (observed with Moonshot
                    # through OpenRouter) reject both null and empty-string
                    # assistant turns. A reasoning-only response has no usable
                    # sibling text/tool payload to continue, so omit it before
                    # the harness sends its no-tool nudge.
                    continue
                # An assistant message with neither text nor tool_calls is
                # rejected by the OpenAI chat API ("content: expected a string,
                # got null"). This happens when a worker's turn produced no
                # visible text and no tool call and the loop re-sends it after a
                # nudge (e.g. a one-shot verify episode). Coerce to "" so the
                # re-send is valid; well-formed messages are unaffected.
                chat_message["content"] = ""
            chat_message.update(provider_reasoning)
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


def _messages_to_openrouter_chat(
    messages: list[dict[str, Any]],
    *,
    system: str = "",
) -> list[dict[str, Any]]:
    """OpenRouter chat conversion with cross-provider compatibility guards."""
    return _messages_to_openai_chat(
        messages,
        system=system,
        passthrough_provider="openrouter",
        drop_empty_assistant=True,
    )


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


def _openrouter_chat_response_to_model_response(response: Any) -> ModelResponse:
    """Normalize OpenRouter chat while retaining re-sendable reasoning data."""
    normalized = _openai_chat_response_to_model_response(response)
    message = response.choices[0].message
    reasoning_item: dict[str, Any] = {}

    reasoning_details = getattr(message, "reasoning_details", None)
    if reasoning_details:
        reasoning_item["reasoning_details"] = [
            _output_item_to_dict(item) for item in reasoning_details
        ]
    reasoning = (
        getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
    )
    if isinstance(reasoning, str) and reasoning:
        reasoning_item["reasoning"] = reasoning

    if not reasoning_item:
        return normalized
    return ModelResponse(
        content=[
            *normalized.content,
            ProviderExtraBlock(
                provider="openrouter",
                kind="reasoning",
                items=[reasoning_item],
            ),
        ],
        usage=normalized.usage,
        raw=normalized.raw,
    )


# ---------------------------------------------------------------------------
# OpenAI Responses API (/v1/responses)
# ---------------------------------------------------------------------------
#
# GPT-5.6 reasoning tiers reject function tools on /v1/chat/completions unless
# reasoning_effort is 'none'.  The Responses API accepts function tools with the
# model's default reasoning effort, so these helpers mirror the chat converters
# above using the flat Responses item shapes:
#   - tools:               {"type": "function", "name", "description", "parameters"}
#   - assistant tool call: {"type": "function_call", "call_id", "name", "arguments"}
#   - tool result:         {"type": "function_call_output", "call_id", "output"}
#   - text message:        {"role", "content"}


def _requires_responses_api(model: str) -> bool:
    """Return True when a model should be driven through /v1/responses.

    ``DECIPHER_OPENAI_API=responses|chat`` forces either path for any model
    (for experiments and future tiers); otherwise model ids beginning with
    ``gpt-5.6`` default to the Responses API and everything else (e.g. gpt-5.5)
    stays on chat completions.
    """
    import os

    override = os.environ.get("DECIPHER_OPENAI_API", "").strip().lower()
    if override == "responses":
        return True
    if override == "chat":
        return False
    return (model or "").strip().lower().startswith("gpt-5.6")


def _tools_to_openai_responses(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    out = []
    for tool in tools:
        out.append({
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        })
    return out


# Fields accepted when re-sending a captured reasoning item as a Responses API
# input item.  Captured items are verbatim response dumps and carry
# response-only fields (at minimum ``status``) that the live endpoint rejects
# with 400 ``unknown_parameter`` on input (confirmed against gpt-5.6-sol, even
# though the openai 2.32.0 SDK's ResponseReasoningItemParam TypedDict nominally
# lists ``status``).  An explicit whitelist keeps any future response-only
# fields from recurring.
_REASONING_INPUT_ITEM_FIELDS = ("type", "id", "summary", "encrypted_content", "content")


def _sanitize_reasoning_item_for_input(item: dict[str, Any]) -> dict[str, Any]:
    """Reduce a captured reasoning item to fields valid as request input.

    Keeps whitelisted fields only and drops None-valued keys (e.g. a null
    ``content``), which the API may also reject.  The captured block in the
    message history stays verbatim; this applies at re-emit time only.
    """
    return {
        key: item[key]
        for key in _REASONING_INPUT_ITEM_FIELDS
        if key in item and item[key] is not None
    }


def _messages_to_openai_responses(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Anthropic-style messages to Responses API input items.

    The system prompt is supplied separately via ``instructions=`` on
    ``responses.create`` and is not represented here.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            function_calls: list[dict[str, Any]] = []
            reasoning_items: list[dict[str, Any]] = []
            passback = _reasoning_passback_enabled()
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block_type == "tool_use":
                    function_calls.append({
                        "type": "function_call",
                        "call_id": str(block.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"),
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(block.get("input") or {}),
                    })
                elif (
                    passback
                    and block_type == "provider_extra"
                    and block.get("provider") == "openai"
                    and block.get("kind") == "reasoning"
                ):
                    for item in block.get("items") or []:
                        if isinstance(item, dict):
                            sanitized = _sanitize_reasoning_item_for_input(item)
                            # Require encrypted_content: without it the item
                            # cannot be resolved server-side under store=False
                            # (e.g. it was captured while the passback env flag
                            # was toggled off mid-run), so skip it.
                            if "encrypted_content" in sanitized:
                                reasoning_items.append(sanitized)
            joined = "\n\n".join(t for t in text_parts if t)
            # Reasoning items are native output items that precede their sibling
            # function_call items in the same turn; re-emit them in that order.
            # A reasoning-only turn (no text and no function calls — e.g. from
            # max-token exhaustion) is skipped entirely: a dangling reasoning
            # item with no sibling output item is rejected by the API.
            if joined or function_calls:
                out.extend(reasoning_items)
            if joined:
                out.append({"role": "assistant", "content": joined})
            out.extend(function_calls)
        elif role == "user" and isinstance(content, list):
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    out.append({
                        "type": "function_call_output",
                        "call_id": str(block.get("tool_use_id", "")),
                        "output": str(block.get("content", "")),
                    })
                elif block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
            if text_parts:
                out.append({"role": "user", "content": "\n\n".join(text_parts)})
        else:
            out.append({"role": role, "content": _content_to_text(content)})
    return out


def _openai_responses_response_to_model_response(response: Any) -> ModelResponse:
    content: list[ModelContentBlock] = []
    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None)
        if item_type == "function_call":
            raw_args = getattr(item, "arguments", "{}") or "{}"
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = {}
            content.append(
                ToolUseBlock(
                    id=str(getattr(item, "call_id", "")),
                    name=str(getattr(item, "name", "")),
                    input=parsed if isinstance(parsed, dict) else {},
                )
            )
        elif item_type == "message":
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "output_text":
                    content.append(TextBlock(text=str(getattr(part, "text", ""))))
    if _reasoning_passback_enabled():
        # Preserve reasoning output items verbatim (including encrypted_content)
        # as one opaque block so they can be re-sent on subsequent turns.
        reasoning_items = [
            _output_item_to_dict(item)
            for item in getattr(response, "output", None) or []
            if getattr(item, "type", None) == "reasoning"
        ]
        if reasoning_items:
            content.append(
                ProviderExtraBlock(
                    provider="openai",
                    kind="reasoning",
                    items=reasoning_items,
                )
            )
    raw_usage = getattr(response, "usage", None)
    input_details = getattr(raw_usage, "input_tokens_details", None)
    usage = ModelUsage(
        input_tokens=int(getattr(raw_usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw_usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(input_details, "cached_tokens", 0) or 0),
    )
    return ModelResponse(content=content, usage=usage, raw=response)


def _output_item_to_dict(item: Any) -> dict[str, Any]:
    """Best-effort verbatim dict for a Responses API output item.

    Handles real pydantic SDK models (``model_dump``), plain dicts, and the
    ``SimpleNamespace`` fakes used by the test suite.
    """
    if isinstance(item, dict):
        return dict(item)
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:  # noqa: BLE001 - fall through to attribute copy
            dumped = None
        if isinstance(dumped, dict):
            return dumped
    if hasattr(item, "__dict__"):
        return dict(vars(item))
    return {}


def _strip_provider_extra_blocks(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop ``provider_extra`` passthrough blocks from message content.

    Providers other than the one that emitted them (e.g. the Anthropic messages
    API) do not understand these opaque blocks and would reject them, so they
    are removed before the request is sent.  Messages without such blocks are
    returned unchanged (same object).
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "provider_extra"
            for block in content
        ):
            filtered = [
                block
                for block in content
                if not (isinstance(block, dict) and block.get("type") == "provider_extra")
            ]
            out.append({**message, "content": filtered})
        else:
            out.append(message)
    return out


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
