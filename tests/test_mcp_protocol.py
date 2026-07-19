"""MCP stdio protocol — scripted subprocess client (Part 11.1).

Spawns `python -m mcp_server` and drives it over real pipes; zero API spend
(verify provider forced to none).
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_SRC = os.path.join(_ROOT, "src")
_TIMEOUT = 30.0


class StdioClient:
    def __init__(self, tmp_registry: str):
        env = {**os.environ, "PYTHONPATH": _SRC}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "mcp_server", "--registry-dir", tmp_registry,
             "--verify-provider", "none"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, cwd=_ROOT,
        )
        self.lines: list[str] = []

    def send_raw(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def send(self, obj: dict) -> None:
        self.send_raw(json.dumps(obj))

    def read(self) -> dict:
        ready, _, _ = select.select([self.proc.stdout], [], [], _TIMEOUT)
        if not ready:
            raise AssertionError("timed out waiting for a response line")
        line = self.proc.stdout.readline()
        self.lines.append(line)
        return json.loads(line)

    def request(self, obj: dict) -> dict:
        self.send(obj)
        return self.read()

    def close(self) -> int:
        self.proc.stdin.close()
        return self.proc.wait(timeout=_TIMEOUT)


@pytest.fixture
def client(tmp_path):
    c = StdioClient(str(tmp_path / "reg"))
    try:
        yield c
    finally:
        try:
            if c.proc.poll() is None:
                c.proc.stdin.close()
                c.proc.wait(timeout=_TIMEOUT)
        except Exception:
            c.proc.kill()


def _init(client, version="2025-06-18"):
    return client.request({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": version, "clientInfo": {"name": "cc"}},
    })


def test_initialize(client):
    r = _init(client)
    assert r["result"]["protocolVersion"] == "2025-06-18"
    assert r["result"]["serverInfo"]["name"] == "decipher"
    assert "tools" in r["result"]["capabilities"]


def test_unsupported_version_falls_back(client):
    r = _init(client, version="1999-01-01")
    assert r["result"]["protocolVersion"] == "2025-06-18"


def test_notification_no_response(client):
    _init(client)
    # A notification produces no response line; the next request's response is
    # the ping, proving no stray line was emitted for the notification.
    client.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    r = client.request({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert r["id"] == 2 and r["result"] == {}


def test_tools_list(client):
    _init(client)
    r = client.request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = r["result"]["tools"]
    assert len(tools) == 23
    for t in tools:
        assert set(t) >= {"name", "description", "inputSchema"}
    names = {t["name"] for t in tools}
    assert "investigation_start" in names and "meta_declare_solution" in names
    start = next(t for t in tools if t["name"] == "investigation_start")
    assert "ciphertext" in start["inputSchema"]["required"]


def test_tools_call_ok(client):
    _init(client)
    r = client.request({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "investigation_list", "arguments": {}}})
    assert r["result"]["isError"] is False
    body = json.loads(r["result"]["content"][0]["text"])
    assert body["investigations"] == []


def test_tools_call_unknown_is_error(client):
    _init(client)
    r = client.request({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "does_not_exist", "arguments": {}}})
    assert r["result"]["isError"] is True


def test_malformed_json_then_recovers(client):
    _init(client)
    client.send_raw("{ this is not json")
    r = client.read()
    assert r["error"]["code"] == -32700
    assert r["id"] is None
    # server still answers afterwards
    r = client.request({"jsonrpc": "2.0", "id": 3, "method": "ping"})
    assert r["result"] == {}


def test_unknown_method_with_id(client):
    _init(client)
    r = client.request({"jsonrpc": "2.0", "id": 2, "method": "bogus/method"})
    assert r["error"]["code"] == -32601


def test_stdout_purity(client):
    _init(client)
    client.request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    client.request({"jsonrpc": "2.0", "id": 3, "method": "ping"})
    # Every line the server emitted parses as JSON.
    for line in client.lines:
        json.loads(line)


def test_eof_exits_clean(client):
    _init(client)
    assert client.close() == 0


def test_investigation_list_reports_server_code(client):
    _init(client)
    r = client.request({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "investigation_list", "arguments": {}}})
    body = json.loads(r["result"]["content"][0]["text"])
    sc = body["server_code"]
    assert set(sc) >= {"git_head", "dirty", "started_at"}
    # dev/CI checkouts are git repos; the value is the live process's code rev
    assert sc["git_head"], "expected a short git head in a git checkout"
    assert isinstance(sc["dirty"], bool)
