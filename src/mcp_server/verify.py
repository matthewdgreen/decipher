"""request_independent_verification glue + provider resolution (Part 7).

Provider resolution happens ONCE at server start; the provider object is passed
to every runtime as the host's ``model_provider``. The credential boundary
holds by construction: the key lives only inside the provider object in the
server process, and no tool result ever contains it.
"""
from __future__ import annotations

import json
import sys
from typing import Any


def resolve_verify_provider(
    verify_provider_flag: str, verify_model_flag: str | None
) -> tuple[Any, str | None]:
    """Resolve ``(provider_object, model_name)`` for server-side verify episodes.

    ``none`` forces keyless; ``auto`` takes the first provider with a key;
    an explicit provider warns to stderr and degrades keyless when its key is
    absent (the server still boots — absent keys never block installation).
    """
    from agent.model_provider import default_model_for_provider, make_model_provider
    from cli import _probe_api_key

    flag = (verify_provider_flag or "auto").strip().lower()
    if flag == "none":
        return (None, None)

    if flag == "auto":
        provider = None
        key = ""
        for candidate in ("anthropic", "openai", "gemini", "openrouter"):
            candidate_key = _probe_api_key(candidate)
            if candidate_key:
                provider, key = candidate, candidate_key
                break
        if provider is None:
            return (None, None)
    else:
        provider = flag
        key = _probe_api_key(provider)
        if not key:
            print(
                f"[decipher-mcp] verify provider {provider!r} has no API key; "
                "verification unavailable (keyless mode).",
                file=sys.stderr,
            )
            return (None, None)

    model = verify_model_flag or default_model_for_provider(provider)
    return (make_model_provider(provider=provider, api_key=key, model=model), model)


def run_independent_verification(
    runtime: Any, verification_available: bool, branch: str
) -> tuple[dict, bool]:
    """Run one server-side independent verification (Part 7.2).

    Returns ``(body, should_commit)``. The two typed early exits
    (no provider / cost ceiling) change no state and skip the commit.
    """
    host = runtime.host
    if not verification_available:
        return ({
            "status": "unavailable",
            "reason": "no_verification_provider",
            "branch": branch,
            "declaration_gate": "closed",
            "note": (
                "The server has no API key for verify episodes. The declaration "
                "gate (DECL-1) remains closed; work is preserved. Add a key "
                "(OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / "
                "OPENROUTER_API_KEY, .decipher_keys/<provider>_api_key, or macOS "
                "keychain service 'decipher'), restart the MCP server, and rerun "
                "this tool."
            ),
        }, False)
    if host.cost_ceiling_reached():
        return ({
            "status": "blocked",
            "reason": "cost_ceiling_reached",
            "committed_cost_usd": host.committed_cost(),
            "max_cost_usd": host.max_cost_usd,
        }, False)
    payload_str = host._dispatch_verify_run(
        {"branches": [branch], "goal": ""}, runtime.state.turn
    )
    payload = json.loads(payload_str)
    host.sync_budget()
    return (payload, True)
