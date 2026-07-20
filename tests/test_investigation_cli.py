"""Tests for the structured investigation CLI (milestone I-1).

Covers the sub-spec's six-test list: manifest auto-registration parity, E2E
reads with body parity against the shared service, the `call` escape hatch
(incl. the read-only guard), the input-mode contract, the exit-code table, and
stdout purity. State is seeded by calling ``InvestigationService.dispatch`` with
a short ciphertext directly (synchronous, $0, no provider); the CLI is invoked
in-process (argv → parser → ``run_investigation_command``).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pytest

import investigation_cli as icli
from investigation_service import manifest
from investigation_service.service import InvestigationService, LeasePolicy
from mcp_server.registry import InvestigationRegistry


_CIPHERTEXT = "HELLO WORLD FROM THE INVESTIGATION CLI TEST SEED CORPUS"

# The nine read-class friendly verbs (sub-spec §0 table). Pinned by hand so the
# parity test asserts the manifest and the registration BOTH match this set.
_EXPECTED_READ_VERBS = {
    "list", "status", "overview", "diagnose", "decode",
    "next-steps", "candidates", "candidate", "adjudicate",
}


# --------------------------------------------------------------------------- #
# Harness                                                                      #
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    """A parser shaped like ``cli.main()``'s (top-level `command` subparsers)."""
    parser = argparse.ArgumentParser(prog="decipher")
    subparsers = parser.add_subparsers(dest="command")
    icli.add_investigation_subparser(subparsers)
    return parser


def _seed(registry_dir: Path) -> str:
    """Create one investigation via the shared service; return its id."""
    service = InvestigationService(
        registry=InvestigationRegistry(registry_dir),
        client_name="test",
        lease_policy=LeasePolicy.SESSION_HELD,
    )
    result = service.dispatch(
        "investigation_start", {"ciphertext": _CIPHERTEXT, "language": "en"}
    )
    return result["investigation_id"]


def _direct(registry_dir: Path, name: str, arguments: dict) -> dict:
    """Dispatch an operation directly through a fresh service (parity oracle)."""
    service = InvestigationService(
        registry=InvestigationRegistry(registry_dir),
        client_name="cli",
        lease_policy=LeasePolicy.SESSION_HELD,
    )
    return service.dispatch(name, arguments)


def _run(registry_dir: Path, verb_args: list[str], capsys) -> tuple[int, dict, str]:
    """Invoke the CLI in-process; return (exit_code, parsed_body, raw_stdout)."""
    argv = ["investigation", "--registry-dir", str(registry_dir), *verb_args]
    args = _build_parser().parse_args(argv)
    code = icli.run_investigation_command(args)
    captured = capsys.readouterr()
    raw = captured.out
    # stdout purity: exactly one JSON document terminated by a single newline.
    assert raw.endswith("\n")
    body = json.loads(raw[:-1])  # json.loads consumes it fully after the newline
    return code, body, raw


def _tree_snapshot(root: Path) -> dict[str, str]:
    """Map every file under root to a content hash (for untouched-dir asserts)."""
    snap: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            snap[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return snap


# --------------------------------------------------------------------------- #
# 1. Auto-registration parity                                                  #
# --------------------------------------------------------------------------- #
def test_auto_registration_parity():
    # (a) the manifest's read-class cli_verb set is exactly the nine.
    manifest_read_verbs = {
        op.cli_verb for op in manifest.OPERATIONS if op.operation_class == "read"
    }
    assert manifest_read_verbs == _EXPECTED_READ_VERBS

    # (b) the actually-registered friendly verbs (minus reserved `call`) match.
    parser = _build_parser()
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    inv = subparsers_action.choices["investigation"]
    verb_action = next(
        a for a in inv._actions if isinstance(a, argparse._SubParsersAction)
    )
    registered = set(verb_action.choices)
    assert "call" in registered  # reserved transport verb is present
    assert registered - {"call"} == _EXPECTED_READ_VERBS


# --------------------------------------------------------------------------- #
# 2. E2E reads + body parity                                                   #
# --------------------------------------------------------------------------- #
def test_e2e_reads_body_parity(tmp_path, capsys):
    iid = _seed(tmp_path)

    # list finds the seeded id.
    code, body, _ = _run(tmp_path, ["list"], capsys)
    assert code == 0
    ids = {inv["investigation_id"] for inv in body["investigations"]}
    assert iid in ids
    assert body == _direct(tmp_path, "investigation_list", {})

    # status ID
    code, body, _ = _run(tmp_path, ["status", iid], capsys)
    assert code == 0
    assert body == _direct(tmp_path, "investigation_status", {"investigation_id": iid})

    # decode ID --branch main
    code, body, _ = _run(tmp_path, ["decode", iid, "--branch", "main"], capsys)
    assert code == 0
    assert body == _direct(
        tmp_path, "decode_show", {"investigation_id": iid, "branch": "main"}
    )

    # candidates ID
    code, body, _ = _run(tmp_path, ["candidates", iid], capsys)
    assert code == 0
    assert body == _direct(tmp_path, "candidate_list", {"investigation_id": iid})


# --------------------------------------------------------------------------- #
# 3. call escape hatch                                                         #
# --------------------------------------------------------------------------- #
def test_call_read_matches_friendly(tmp_path, capsys):
    iid = _seed(tmp_path)
    _, friendly_body, _ = _run(tmp_path, ["status", iid], capsys)
    _, call_body, _ = _run(
        tmp_path,
        ["call", "investigation_status", "--input-json", json.dumps({"investigation_id": iid})],
        capsys,
    )
    assert call_body == friendly_body


def test_call_mutating_op_not_yet_available(tmp_path, capsys):
    _seed(tmp_path)
    before = _tree_snapshot(tmp_path)
    code, body, _ = _run(
        tmp_path, ["call", "meta_declare_solution", "--input-json", "{}"], capsys
    )
    assert code == 2
    assert body["status"] == "error"
    assert body["reason"] == "operation_not_yet_available"
    # The registry directory is untouched: no lease created, no event appended.
    assert _tree_snapshot(tmp_path) == before


def test_call_unknown_operation(tmp_path, capsys):
    _seed(tmp_path)
    code, body, _ = _run(tmp_path, ["call", "nope_op"], capsys)
    assert code == 2
    assert body["reason"] == "unknown_operation"


# --------------------------------------------------------------------------- #
# 4. Input modes                                                               #
# --------------------------------------------------------------------------- #
def test_input_mode_duplicate_id_conflict(tmp_path, capsys):
    iid = _seed(tmp_path)
    code, body, _ = _run(
        tmp_path,
        ["status", iid, "--input-json", json.dumps({"investigation_id": iid})],
        capsys,
    )
    assert code == 2
    assert body["reason"] == "invalid_cli_arguments"


def test_input_mode_malformed_json(tmp_path, capsys):
    _seed(tmp_path)
    code, body, _ = _run(tmp_path, ["status", "--input-json", "{not json"], capsys)
    assert code == 2
    assert body["reason"] == "invalid_cli_arguments"


def test_input_mode_stdin_input_file(tmp_path, capsys, monkeypatch):
    iid = _seed(tmp_path)
    monkeypatch.setattr(
        icli.sys, "stdin", io.StringIO(json.dumps({"investigation_id": iid}))
    )
    code, body, _ = _run(tmp_path, ["status", "--input-file", "-"], capsys)
    assert code == 0
    assert body["investigation_id"] == iid
    assert body == _direct(tmp_path, "investigation_status", {"investigation_id": iid})


def test_input_mode_id_plus_json_merge(tmp_path, capsys):
    # Positional ID + --input-json with NO duplicate id merges and succeeds (§2).
    iid = _seed(tmp_path)
    code, body, _ = _run(
        tmp_path,
        ["decode", iid, "--input-json", json.dumps({"branch": "main"})],
        capsys,
    )
    assert code == 0
    assert body == _direct(
        tmp_path, "decode_show", {"branch": "main", "investigation_id": iid}
    )


# --------------------------------------------------------------------------- #
# 5. Exit-code table                                                           #
# --------------------------------------------------------------------------- #
def test_exit_unknown_investigation_id(tmp_path, capsys):
    _seed(tmp_path)
    code, body, _ = _run(tmp_path, ["status", "does-not-exist"], capsys)
    assert code == 1  # any non-invalid_arguments domain error → 1
    assert body["reason"] == "investigation_not_found"  # service's own string


def test_exit_decode_missing_required_branch(tmp_path, capsys):
    # decode_show requires `branch`, but the CLI does NOT make friendly flags
    # argparse-required (so --input-json remains usable). The missing branch is
    # therefore caught by the service's schema validation, not argparse:
    # the invalid_arguments path → exit 2.
    iid = _seed(tmp_path)
    code, body, _ = _run(tmp_path, ["decode", iid], capsys)
    assert code == 2
    assert body["status"] == "error"
    assert body["reason"] == "invalid_arguments"


def test_exit_internal_error(tmp_path, capsys, monkeypatch):
    iid = _seed(tmp_path)

    def _boom(self, name, arguments):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(InvestigationService, "dispatch", _boom)
    code, body, raw = _run(tmp_path, ["status", iid], capsys)
    assert code == 5
    assert body == {"status": "error", "reason": "internal_error"}
    # No traceback leaks to stdout (only the single JSON object).
    assert raw.strip() == json.dumps(body, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 6. stdout purity (exercised by _run's assertions across every case above,    #
#    plus an explicit single-document check here).                            #
# --------------------------------------------------------------------------- #
def test_stdout_is_single_json_document(tmp_path, capsys):
    iid = _seed(tmp_path)
    argv = ["investigation", "--registry-dir", str(tmp_path), "overview", iid]
    args = _build_parser().parse_args(argv)
    icli.run_investigation_command(args)
    raw = capsys.readouterr().out
    # Exactly one JSON document: decoder consumes the whole payload after the
    # single trailing newline, with nothing left over.
    assert raw.count("\n") == 1
    doc = json.loads(raw[:-1])
    assert isinstance(doc, dict)
