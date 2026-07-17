"""MCP stdio JSON-RPC framing + method dispatch (Part 2).

Pure stdlib (``json``, ``sys``, ``traceback``). No new dependency: the
tools-only MCP subset is three request methods plus notifications over
newline-delimited JSON-RPC 2.0. One JSON message per line, UTF-8, no
Content-Length headers.

The server writes NOTHING to stdout except protocol frames; diagnostics go to
stderr.
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any

# Protocol versions this server understands. If the client requests one of
# these, echo it back; otherwise fall back to the newest.
SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
DEFAULT_PROTOCOL_VERSION = "2025-06-18"


def _error(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def _result(mid: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _write(stdout, obj: dict[str, Any]) -> None:
    stdout.write(json.dumps(obj) + "\n")
    stdout.flush()


def dispatch(handler: "Any", message: dict[str, Any]) -> dict[str, Any] | None:
    """Route one well-formed JSON-RPC message to the handler.

    Returns the response dict for a REQUEST (message with an ``id``), or
    ``None`` for a NOTIFICATION (no ``id``) — the caller writes only for
    requests.
    """
    method = message.get("method")
    mid = message.get("id")
    is_request = "id" in message
    params = message.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    if method == "initialize":
        return _result(mid, handler.initialize(params)) if is_request else None
    if method == "ping":
        return _result(mid, {}) if is_request else None
    if method == "tools/list":
        return _result(mid, {"tools": handler.tool_list()}) if is_request else None
    if method == "tools/call":
        if not is_request:
            return None
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        text, is_error = handler.call_tool(name, arguments)
        return _result(mid, {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        })
    if isinstance(method, str) and method.startswith("notifications/"):
        # No-op notification (e.g. notifications/initialized). Never answered.
        return None
    # Unknown method: an error for a request; silent for a notification.
    return _error(mid, -32601, "method not found") if is_request else None


def serve_loop(handler: "Any", stdin=None, stdout=None) -> None:
    """Read newline-delimited JSON-RPC from ``stdin`` and answer on ``stdout``.

    One bad call never kills the server: parse failures and handler exceptions
    become JSON-RPC errors (or are dropped for notifications); the loop
    continues. EOF (client closed stdin) returns cleanly so the process exits 0.
    """
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            _write(stdout, _error(None, -32700, "parse error"))
            continue
        if not isinstance(message, dict) or "method" not in message:
            mid = message.get("id") if isinstance(message, dict) else None
            _write(stdout, _error(mid, -32600, "invalid request"))
            continue
        is_request = "id" in message
        try:
            response = dispatch(handler, message)
        except Exception:  # noqa: BLE001 - one bad call must not kill the server
            traceback.print_exc(file=sys.stderr)
            if is_request:
                _write(stdout, _error(message.get("id"), -32603, "internal error"))
            continue
        if is_request and response is not None:
            _write(stdout, response)
    return
