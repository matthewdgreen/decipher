"""DecipherMCPServer: JSON-RPC framing + session lifecycle over the service.

One process = one server = one registry. The server is now a thin JSON-RPC skin:
it owns protocol framing (initialize / tools list / tools call), the session's
client identity, and lifecycle finalize. All domain work — the dispatch pipeline
(validate, resolve, terminal-check, lease, revision-check, dispatch, commit,
revision-inject) and every gate — lives in the transport-neutral
:class:`~investigation_service.service.InvestigationService` (interface lockstep
contract, spec §6). The CLI is a second skin over the same service. MCP passes
the ``SESSION_HELD`` lease policy so runtimes and their writer leases persist for
the process lifetime, exactly as before.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from investigation_service.service import InvestigationService, LeasePolicy

from mcp_server import protocol, tools, verify
from mcp_server.registry import (
    InvestigationRegistry,
    default_registry_dir,
)


class DecipherMCPServer:
    """The handler the protocol loop dispatches into (a skin over the service)."""

    def __init__(
        self,
        *,
        registry: InvestigationRegistry,
        verify_provider: Any = None,
        verify_model: str | None = None,
        max_cost_usd: float = 5.0,
        synchronous_experiments: bool = False,
    ) -> None:
        self.service = InvestigationService(
            registry=registry,
            verify_provider=verify_provider,
            verify_model=verify_model,
            max_cost_usd=max_cost_usd,
            synchronous_experiments=synchronous_experiments,
            lease_policy=LeasePolicy.SESSION_HELD,
        )

    # Session-lifecycle attributes read on the server object by the registry and
    # tool test suites; delegated to the service that owns them.
    @property
    def registry(self) -> InvestigationRegistry:
        return self.service.registry

    @property
    def _runtimes(self) -> dict:
        return self.service._runtimes

    # ---------------------------------------------------------------- protocol
    def initialize(self, params: dict) -> dict:
        requested = params.get("protocolVersion")
        version = (
            requested if requested in protocol.SUPPORTED_PROTOCOL_VERSIONS
            else protocol.DEFAULT_PROTOCOL_VERSION
        )
        client_info = params.get("clientInfo") or {}
        self.service.client_name = str(client_info.get("name") or "unknown")
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "decipher", "version": "0.1.0"},
        }

    def tool_list(self) -> list[dict]:
        return tools.render_tool_list()

    def call_tool(self, name: Any, arguments: dict) -> tuple[str, bool]:
        if name not in tools.TOOL_NAMES:
            return (json.dumps({"error": f"unknown tool: {name}"}), True)
        try:
            body = self.service.dispatch(str(name), arguments or {})
        except Exception:  # noqa: BLE001 - a handler crash is a protocol error
            traceback.print_exc(file=sys.stderr)
            return (json.dumps({"error": f"internal error running {name}"}), True)
        return (json.dumps(body, ensure_ascii=False), False)

    # ---------------------------------------------------------------- lifecycle
    def shutdown(self) -> None:
        """Finalize + commit every leased runtime (Part 4.4)."""
        self.service.shutdown()


def serve_stdio(
    *,
    registry_dir: str | None = None,
    verify_provider: str = "auto",
    verify_model: str | None = None,
    max_cost_usd: float = 5.0,
    synchronous_experiments: bool = False,
) -> None:
    """Boot the stdio server. Nothing but protocol frames reaches stdout."""
    root = Path(registry_dir).expanduser() if registry_dir else default_registry_dir()
    provider, model = verify.resolve_verify_provider(verify_provider, verify_model)
    server = DecipherMCPServer(
        registry=InvestigationRegistry(root),
        verify_provider=provider, verify_model=model,
        max_cost_usd=max_cost_usd, synchronous_experiments=synchronous_experiments,
    )
    if provider is None:
        print("[decipher-mcp] verify provider: none (verification unavailable)", file=sys.stderr)
    else:
        print(f"[decipher-mcp] verify provider: {model} (verification available)", file=sys.stderr)
    try:
        protocol.serve_loop(server)
    finally:
        server.shutdown()
