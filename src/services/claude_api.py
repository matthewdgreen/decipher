from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any, Generator

import anthropic


class ClaudeAPIError(Exception):
    pass


# Pricing per million tokens (MTok).  Cache-read tokens are billed at ~10% of
# the standard input rate.  Figures are approximate; update as needed.
_PRICING: dict[str, tuple[float, float, float]] = {
    # model_prefix: (input $/MTok, output $/MTok, cache_read $/MTok)
    "claude-opus-4":     (15.00, 75.00, 1.50),
    "claude-sonnet-4":   ( 3.00, 15.00, 0.30),
    "claude-haiku-4":    ( 0.80,  4.00, 0.08),
    "claude-opus-3":     (15.00, 75.00, 1.50),
    "claude-sonnet-3":   ( 3.00, 15.00, 0.30),
    "claude-haiku-3":    ( 0.25,  1.25, 0.03),
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Return estimated USD cost for a set of token counts.

    Matches model name against _PRICING prefixes (longest-first).
    Returns 0.0 if no matching entry is found.
    """
    prefix = ""
    for p in sorted(_PRICING, key=len, reverse=True):
        if model.startswith(p):
            prefix = p
            break
    if not prefix:
        return 0.0
    inp_rate, out_rate, cache_rate = _PRICING[prefix]
    billed_input = max(0, input_tokens - cache_read_tokens)
    return (
        billed_input * inp_rate / 1_000_000
        + cache_read_tokens * cache_rate / 1_000_000
        + output_tokens * out_rate / 1_000_000
    )


class ClaudeAPI:
    """Client for the Anthropic Claude API (messages + vision + tool use)."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=3,
            timeout=120.0,
        )

    def send_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 4096,
    ) -> anthropic.types.Message:
        """Send a single-turn message, optionally with tools."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        return self._create_with_retry(kwargs)

    def _create_with_retry(
        self,
        kwargs: dict[str, Any],
        max_retries: int = 4,
        base_delay: float = 60.0,
    ) -> anthropic.types.Message:
        """Call messages.create with exponential backoff on 429 rate-limit errors."""
        delay = base_delay
        for attempt in range(max_retries + 1):
            try:
                return self.client.messages.create(**kwargs)
            except anthropic.RateLimitError as e:
                if attempt == max_retries:
                    raise ClaudeAPIError(str(e)) from e
                print(f"\n  [rate-limit] waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}…",
                      flush=True)
                time.sleep(delay)
                delay = min(delay * 1.5, 300.0)
            except anthropic.APIError as e:
                raise ClaudeAPIError(str(e)) from e

    def send_message_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 4096,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream a message response. Yields event dicts.

        Event types:
          {"type": "text_delta", "text": "..."}
          {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
          {"type": "message_complete", "message": Message}
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        try:
            with self.client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                yield {"type": "text_delta", "text": event.delta.text}
                        elif event.type == "content_block_stop":
                            pass
                final = stream.get_final_message()
                yield {"type": "message_complete", "message": final}
        except anthropic.APIError as e:
            raise ClaudeAPIError(str(e)) from e

    def vision_request(self, image_path: str, prompt: str) -> str:
        """Send an image to Claude Vision and return the text response."""
        path = Path(image_path)
        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        image_data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return message.content[0].text
