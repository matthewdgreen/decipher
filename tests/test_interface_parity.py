"""I-6 manifest projection/input contract; lifecycle tests live in test_investigation_cli."""
from __future__ import annotations

import argparse
import json

import pytest

import investigation_cli as cli
from investigation_service import manifest
from mcp_server.registry import InvestigationRegistry
from mcp_server.server import DecipherMCPServer


def parser():
    result = argparse.ArgumentParser()
    cli.add_investigation_subparser(result.add_subparsers(dest="command"))
    return result


@pytest.mark.parametrize("op", manifest.OPERATIONS, ids=lambda op: op.name)
@pytest.mark.parametrize("mode", ["inline", "file", "stdin"])
def test_every_operation_preserves_canonical_json(op, mode, tmp_path, monkeypatch):
    import io

    # Transport fidelity only: domain validation is deliberately downstream.
    payload = {"investigation_id": "fixture", "expected_revision": 7,
               "nested": {"commas": "one,two", "unicode": "ü", "bool": False,
                          "list": [None, {"text": "line\nbreak"}]}}
    encoded = json.dumps(payload)
    if mode == "inline":
        flags = ["--input-json", encoded]
    elif mode == "file":
        target = tmp_path / "input.json"
        target.write_text(encoded)
        flags = ["--input-file", str(target)]
    else:
        monkeypatch.setattr("sys.stdin", io.StringIO(encoded))
        flags = ["--input-file", "-"]
    args = parser().parse_args(["investigation", op.cli_verb, *flags])
    assert cli._build_arguments(args) == (op.name, payload)


def test_new_manifest_operation_appears_on_both_surfaces(tmp_path, monkeypatch):
    dummy = manifest.OperationSpec(
        name="future_read", cli_verb="future-read", description="Future read operation.",
        operation_class="read", external_effect=manifest.ExternalEffect.NEVER,
        schema={"type": "object", "properties": {"payload": {"type": "object"}}},
    )
    # Append only to the canonical list: no CLI/MCP registration changes.
    monkeypatch.setattr(manifest, "OPERATIONS", [*manifest.OPERATIONS, dummy])
    server = DecipherMCPServer(registry=InvestigationRegistry(tmp_path))
    try:
        listed = next(row for row in server.tool_list() if row["name"] == dummy.name)
        assert listed["inputSchema"] == dummy.schema
        args = parser().parse_args([
            "investigation", "future-read", "--payload-json", '{"nested":[1,false,null]}',
        ])
        assert cli._build_arguments(args) == (dummy.name, {"payload": {"nested": [1, False, None]}})
    finally:
        server.service.shutdown()


def test_cli_mcp_cli_continuation_uses_real_transport_adapters(tmp_path, capsys):
    from tests.test_investigation_cli import _run

    code, started, _ = _run(tmp_path, ["start", "--ciphertext", "ABC DEF GHI",
                                     "--language", "en"], capsys)
    assert code == 0
    iid = started["investigation_id"]
    server = DecipherMCPServer(registry=InvestigationRegistry(tmp_path))
    try:
        raw, failed = server.call_tool("hypothesis_branch_create", {
            "investigation_id": iid, "expected_revision": started["revision"],
            "new_name": "from_mcp", "cipher_mode": "mono", "rationale": "transport test",
        })
        created = json.loads(raw)
        assert not failed and created["status"] == "ok"
        raw, failed = server.call_tool("decode_show", {"investigation_id": iid, "branch": "from_mcp"})
        assert not failed
        mcp_decode = json.loads(raw)
        code, blocked, _ = _run(tmp_path, ["branch-create", iid,
            "--revision", str(created["revision"]), "--new-name", "from_cli",
            "--cipher-mode", "mono", "--rationale", "transport test"], capsys)
        assert code == 3 and blocked["reason"] == "writer_lease_held"
    finally:
        server.shutdown()
        server.registry.release_lease(iid)
    code, status, _ = _run(tmp_path, ["status", iid], capsys)
    assert code == 0
    code, decoded, _ = _run(tmp_path, ["decode", iid, "--branch", "from_mcp"], capsys)
    assert code == 0 and decoded == mcp_decode
    code, updated, _ = _run(tmp_path, ["branch-create", iid,
        "--revision", str(status["revision"]), "--new-name", "from_cli",
        "--cipher-mode", "mono", "--rationale", "transport test"], capsys)
    assert code == 0 and updated["status"] == "ok"


@pytest.mark.parametrize("tail", [
    [], ["not-a-verb"], ["status", "abc", "--unknown"],
    ["branch-create", "abc", "--revision", "nope"],
    ["branch-create", "abc", "--revision"],
    ["start", "--format", "not-a-format"],
    ["--verify-provider", "not-a-provider", "verify"],
    ["--max-cost-usd", "not-a-number", "verify"],
    ["experiment-submit", "abc", "--wait", "--detach"],
])
def test_actual_entrypoint_parser_failures_are_single_json(tail, tmp_path):
    import subprocess
    import sys

    registry = tmp_path / "untouched"
    result = subprocess.run([
        sys.executable, "-m", "cli", "investigation", "--registry-dir", str(registry), *tail,
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result
    assert result.stdout.count("\n") == 1
    body = json.loads(result.stdout)
    assert body["status"] == "error" and body["reason"] == "invalid_cli_arguments"
    assert body["detail"]
    assert not registry.exists()


def test_help_and_other_commands_keep_human_argparse_behavior():
    import subprocess
    import sys

    help_result = subprocess.run([sys.executable, "-m", "cli", "investigation", "--help"],
                                 capture_output=True, text=True, timeout=30)
    assert help_result.returncode == 0 and help_result.stdout.startswith("usage:")
    other = subprocess.run([sys.executable, "-m", "cli", "diagnose", "--not-an-option"],
                           capture_output=True, text=True, timeout=30)
    assert other.returncode == 2 and other.stdout == "" and "usage:" in other.stderr


@pytest.mark.parametrize("mode", ["cli_dispatch", "mcp_dispatch", "wait", "detach"])
@pytest.mark.parametrize("terminal", ["solved", "unsolved"])
def test_terminal_transition_between_read_and_lease_blocks_every_writer(mode, terminal, tmp_path, monkeypatch):
    from investigation_service.service import InvestigationService, LeasePolicy
    from tests.test_investigation_cli import _seed

    iid = _seed(tmp_path)
    registry = InvestigationRegistry(tmp_path)
    service = InvestigationService(registry=registry, lease_policy=(
        LeasePolicy.SESSION_HELD if mode == "mcp_dispatch" else LeasePolicy.INVOCATION_HELD
    ))
    real_acquire = registry.acquire_lease
    closed = {}

    def close_between_read_and_acquire(target):
        other = InvestigationRegistry(tmp_path)
        assert other.acquire_lease(target)
        try:
            document = other.load(target)
            document["meta"]["status"] = terminal
            assert other.commit(target, document) == 2
            closed.update(other.load(target))
        finally:
            other.release_lease(target)
        return real_acquire(target)

    def must_not_execute(*_args, **_kwargs):
        raise AssertionError("terminal investigation reached domain execution")

    monkeypatch.setattr(registry, "acquire_lease", close_between_read_and_acquire)
    monkeypatch.setattr(service, "_execute", must_not_execute)
    try:
        args = {"investigation_id": iid, "expected_revision": 2}
        notifications = []
        if mode in {"wait", "detach"}:
            result, revision = service.run_experiment_to_completion(
                args | {"type": "automated_solver", "branch": "main", "config": {}},
                on_running=notifications.append if mode == "detach" else None,
            )
            assert revision is None
            if mode == "detach":
                assert notifications == [result]
        else:
            result = service.dispatch("hypothesis_branch_create", args | {
                "new_name": "forbidden", "cipher_mode": "mono", "rationale": "race test",
            })
        assert result == {"status": "blocked", "reason": "investigation_terminal", "terminal_status": terminal}
        assert registry.load(iid) == closed
        if mode != "mcp_dispatch":
            assert not registry.holds_lease(iid)
    finally:
        service.shutdown()
        registry.release_lease(iid)
