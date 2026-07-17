"""Fresh-clone simulation (Part 11.4).

Copies the tracked working tree into a temp "clone", checks the onboarding
files parse, verifies the launcher fails fast without a venv, and boots the
server from the clone source over stdio through to a keyless declaration gate.
Zero API spend.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TIMEOUT = 30.0


def _git_available() -> bool:
    return shutil.which("git") is not None


pytestmark = pytest.mark.skipif(not _git_available(), reason="git unavailable")


def _make_clone(dst: str) -> None:
    out = subprocess.run(
        ["git", "-C", _ROOT, "ls-files", "-z"],
        capture_output=True, check=True,
    ).stdout
    for rel in out.split(b"\x00"):
        if not rel:
            continue
        relpath = rel.decode("utf-8")
        src = os.path.join(_ROOT, relpath)
        if not os.path.isfile(src):
            continue
        target = os.path.join(dst, relpath)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(src, target)


class _Client:
    def __init__(self, proc):
        self.proc = proc

    def request(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()
        ready, _, _ = select.select([self.proc.stdout], [], [], _TIMEOUT)
        if not ready:
            raise AssertionError("timed out waiting for a response line")
        return json.loads(self.proc.stdout.readline())

    def call(self, name, **args):
        r = self.request({"jsonrpc": "2.0", "id": 99, "method": "tools/call",
                          "params": {"name": name, "arguments": args}})
        return json.loads(r["result"]["content"][0]["text"])


def test_fresh_clone_boots_and_gate_holds(tmp_path):
    clone = str(tmp_path / "clone")
    os.makedirs(clone)
    _make_clone(clone)

    # 2. Onboarding files exist and parse.
    mcp_json = json.loads(open(os.path.join(clone, ".mcp.json")).read())
    assert mcp_json["mcpServers"]["decipher"]["args"] == ["scripts/mcp_launch.sh"]
    codex = open(os.path.join(clone, ".codex/config.toml")).read()
    assert "[mcp_servers.decipher]" in codex
    for rel in ("scripts/mcp_launch.sh", "scripts/bootstrap.sh",
                "docs/mcp_onboarding.md", "docs/prompts/decipher-investigate.md"):
        assert os.path.isfile(os.path.join(clone, rel)), rel

    # 3. Launcher degradation: no .venv -> fast nonzero exit with bootstrap_required.
    proc = subprocess.run(
        ["sh", "scripts/mcp_launch.sh"], cwd=clone,
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode != 0
    assert "bootstrap_required" in proc.stderr

    # 4. Server boots from the clone source (decipher code resolves from the
    #    clone; the test venv supplies third-party deps).
    env = {**os.environ, "PYTHONPATH": os.path.join(clone, "src")}
    # Review fix: PROVE imports resolve from the clone, not the dev venv's
    # editable-install .pth (which keeps the dev src on sys.path behind
    # PYTHONPATH). A forgotten `git add` would otherwise silently resolve
    # from the dev tree and keep this test green — the exact regression this
    # test exists to catch.
    probe = subprocess.run(
        [sys.executable, "-c",
         "import mcp_server, investigation, analysis; "
         "print(mcp_server.__file__); print(investigation.__file__); "
         "print(analysis.__file__)"],
        capture_output=True, text=True, timeout=30, env=env, cwd=clone,
    )
    assert probe.returncode == 0, probe.stderr
    for line in probe.stdout.strip().splitlines():
        assert line.startswith(clone), (
            f"module resolved OUTSIDE the clone: {line}"
        )
    server = subprocess.Popen(
        [sys.executable, "-m", "mcp_server", "--registry-dir",
         str(tmp_path / "registry"), "--verify-provider", "none"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, cwd=clone,
    )
    try:
        client = _Client(server)
        init = client.request({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                               "params": {"protocolVersion": "2025-06-18",
                                          "clientInfo": {"name": "cc"}}})
        assert init["result"]["serverInfo"]["name"] == "decipher"
        tools = client.request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert len(tools["result"]["tools"]) == 23

        started = client.call("investigation_start", ciphertext="hello world test")
        iid = started["investigation_id"]
        assert started["revision"] == 1
        status = client.call("investigation_status", investigation_id=iid)
        assert status["brief"]
        assert "Tokens:" in status["brief"] or "Cipher" in status["brief"]
        client.call("decode_show", investigation_id=iid, branch="main")
        decl = client.call(
            "meta_declare_solution", investigation_id=iid, expected_revision=1,
            branch="main", rationale="x", self_confidence=0.9, reading_summary="x",
            further_iterations_helpful=False, further_iterations_note="x",
        )
        assert decl["reason"] == "attestation_required"
    finally:
        server.stdin.close()
        assert server.wait(timeout=_TIMEOUT) == 0
