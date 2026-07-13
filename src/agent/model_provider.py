"""Provider-neutral model interface for agent loops."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
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
        # GPT-5.6 tiers reject function tools on /v1/chat/completions unless
        # reasoning_effort is 'none' (which would lobotomize the model).  Route
        # them through /v1/responses instead.  gpt-5.5 and earlier keep the
        # chat-completions path below unchanged.
        if _requires_responses_api(self.model):
            return self._send_responses(
                messages=messages,
                tools=tools,
                system=system,
                max_tokens=max_tokens,
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
    ) -> ModelResponse:
        """Send one request through the OpenAI Responses API.

        Used for reasoning tiers (e.g. gpt-5.6-*) that reject function tools on
        /v1/chat/completions.  No ``reasoning`` parameter is set, so the model's
        server-side default effort applies.
        """
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=system,
                input=_messages_to_openai_responses(messages),
                tools=_tools_to_openai_responses(tools),
                max_output_tokens=max_tokens,
            )
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
                messages=_messages_to_openai_chat(messages, system=system),
                tools=_tools_to_openai_chat(tools),
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
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
        return "gpt-5.4"
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

        if _OPENROUTER_ALL_MODEL_IDS is None:
            # Disk cache exists but predates the all_model_ids field (old format),
            # or no cache exists yet.  Do a direct fetch to populate IDs; this
            # also rewrites the cache in the new format.
            try:
                fetch_openrouter_pricing(timeout=timeout, cache_path=cache_path)
            except Exception:  # noqa: BLE001
                return True, ""  # Cannot reach OpenRouter — don't block the run.

        if _OPENROUTER_ALL_MODEL_IDS is not None:
            if model in _OPENROUTER_ALL_MODEL_IDS:
                return True, ""
            # Build a helpful suggestion list: models whose ID contains any
            # component of the supplied name (split on "/").
            parts = [p.lower() for p in model.replace(":", "/").split("/") if p]
            suggestions = sorted(
                mid for mid in _OPENROUTER_ALL_MODEL_IDS
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
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_use":
                    function_calls.append({
                        "type": "function_call",
                        "call_id": str(block.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"),
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(block.get("input") or {}),
                    })
            joined = "\n\n".join(t for t in text_parts if t)
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
    raw_usage = getattr(response, "usage", None)
    input_details = getattr(raw_usage, "input_tokens_details", None)
    usage = ModelUsage(
        input_tokens=int(getattr(raw_usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw_usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(input_details, "cached_tokens", 0) or 0),
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
