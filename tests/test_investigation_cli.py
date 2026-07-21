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
import shutil
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
    """Create one investigation via the shared service; return its id.

    Releases the writer lease before returning: a SESSION_HELD ``start`` holds
    it for the process lifetime, which would block an invocation-held CLI
    mutation running later in the SAME test process (flock is per open-file
    description). Seeding should leave nothing held.
    """
    service = InvestigationService(
        registry=InvestigationRegistry(registry_dir),
        client_name="test",
        lease_policy=LeasePolicy.SESSION_HELD,
    )
    result = service.dispatch(
        "investigation_start", {"ciphertext": _CIPHERTEXT, "language": "en"}
    )
    iid = result["investigation_id"]
    service.registry.release_lease(iid)
    return iid


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
    # (a) the manifest's read-class cli_verb set still contains exactly the nine
    # I-1 reads (create + mutate join the surface in I-2 but the reads are fixed).
    manifest_read_verbs = {
        op.cli_verb for op in manifest.OPERATIONS if op.operation_class == "read"
    }
    assert manifest_read_verbs == _EXPECTED_READ_VERBS

    # (b) I-2 registers ONE friendly verb per manifest operation (read + create +
    # mutate), including the three not-yet-dispatchable ops (they parse; only
    # dispatch is short-circuited). The registered set (minus reserved `call`) is
    # exactly the manifest's cli_verb set.
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
    all_manifest_verbs = {op.cli_verb for op in manifest.OPERATIONS}
    assert registered - {"call"} == all_manifest_verbs
    # The three excluded ops are still registered as verbs.
    for excluded_verb in ("experiment-submit", "experiment-collect", "verify"):
        assert excluded_verb in registered


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


def test_call_excluded_op_not_yet_available(tmp_path, capsys):
    # meta_declare_solution is dispatchable in I-2; the still-excluded ops are
    # the two experiment verbs (I-3) and verify (I-5). `call` on their canonical
    # names returns operation_not_yet_available and never touches the registry.
    _seed(tmp_path)
    before = _tree_snapshot(tmp_path)
    code, body, _ = _run(
        tmp_path,
        ["call", "request_independent_verification", "--input-json", "{}"],
        capsys,
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


# =========================================================================== #
# I-2: mutations, invocation-held leases, exit codes 3/4, id containment       #
# =========================================================================== #
_BRANCH_CREATE = [
    "--new-name", "h", "--cipher-mode", "mono", "--rationale", "x",
]

_DECLARE_UNSOLVED = [
    "--rationale", "give up",
    "--branches-considered-json", json.dumps(["main"]),
    "--reading-summary", "none",
    "--no-further-iterations-helpful",
    "--further-iterations-note", "n/a",
]


def _mcp_service(registry_dir: Path) -> InvestigationService:
    """A SESSION_HELD service — the MCP transport, for parity/containment checks."""
    return InvestigationService(
        registry=InvestigationRegistry(registry_dir),
        client_name="cli",
        lease_policy=LeasePolicy.SESSION_HELD,
    )


# --------------------------------------------------------------------------- #
# Lease lifecycle                                                              #
# --------------------------------------------------------------------------- #
def test_cli_mutation_releases_lease(tmp_path, capsys):
    iid = _seed(tmp_path)
    code, body, _ = _run(
        tmp_path, ["branch-create", iid, "--revision", "1", *_BRANCH_CREATE], capsys
    )
    assert code == 0
    assert body.get("status") == "ok"
    assert body["revision"] == 2
    # After the invocation, a fresh registry instance can acquire immediately.
    reg = InvestigationRegistry(tmp_path)
    assert reg.acquire_lease(iid) is True
    reg.release_lease(iid)


def test_two_sequential_cli_mutations(tmp_path, capsys):
    iid = _seed(tmp_path)
    c1, b1, _ = _run(
        tmp_path, ["branch-create", iid, "--revision", "1",
                   "--new-name", "h1", "--cipher-mode", "mono", "--rationale", "x"],
        capsys,
    )
    assert c1 == 0 and b1["revision"] == 2
    c2, b2, _ = _run(
        tmp_path, ["branch-create", iid, "--revision", "2",
                   "--new-name", "h2", "--cipher-mode", "mono", "--rationale", "y"],
        capsys,
    )
    assert c2 == 0 and b2["revision"] == 3


def test_invocation_held_runtimes_empty_after_mutation(tmp_path):
    # White-box the invocation-held finally: the runtime is dropped and the lease
    # released the moment dispatch returns (sub-spec §4).
    iid = _seed(tmp_path)
    reg = InvestigationRegistry(tmp_path)
    svc = InvestigationService(
        registry=reg, client_name="cli", lease_policy=LeasePolicy.INVOCATION_HELD
    )
    body = svc.dispatch("hypothesis_branch_create", {
        "investigation_id": iid, "expected_revision": 1,
        "new_name": "h", "cipher_mode": "mono", "rationale": "x",
    })
    assert body["revision"] == 2
    assert svc._runtimes == {}
    assert reg.held_lease_ids() == []


def test_invocation_held_releases_on_revision_conflict_direct(tmp_path):
    """Review finding #3: the CLI-level conflict test can't isolate dispatch's
    finally (run_investigation_command's shutdown() mop-up would mask a broken
    release). Direct-service variant: a stale expected_revision early-return
    must leave no lease and no runtime BEFORE any shutdown()."""
    iid = _seed(tmp_path)
    reg = InvestigationRegistry(tmp_path)
    svc = InvestigationService(
        registry=reg, client_name="cli", lease_policy=LeasePolicy.INVOCATION_HELD
    )
    body = svc.dispatch("hypothesis_branch_create", {
        "investigation_id": iid, "expected_revision": 999,
        "new_name": "h", "cipher_mode": "mono", "rationale": "x",
    })
    assert body["status"] == "conflict"
    assert body["reason"] == "revision_mismatch"
    assert svc._runtimes == {}
    assert reg.held_lease_ids() == []


# --------------------------------------------------------------------------- #
# Lease collision                                                              #
# --------------------------------------------------------------------------- #
def test_cli_mutation_lease_collision(tmp_path, capsys):
    iid = _seed(tmp_path)
    # A separate registry instance (a live MCP-style session) holds the lease.
    other = InvestigationRegistry(tmp_path)
    assert other.acquire_lease(iid) is True
    try:
        code, body, _ = _run(
            tmp_path, ["branch-create", iid, "--revision", "1", *_BRANCH_CREATE], capsys
        )
        assert code == 3
        assert body["status"] == "blocked"
        assert body["reason"] == "writer_lease_held"
        assert "holder" in body  # holder hint included verbatim
        # The document is unchanged.
        assert InvestigationRegistry(tmp_path).load(iid)["meta"]["revision"] == 1
    finally:
        other.release_lease(iid)


# --------------------------------------------------------------------------- #
# Revision conflict (exit 4) — and the finally releases on the early return    #
# --------------------------------------------------------------------------- #
def test_cli_mutation_revision_conflict(tmp_path, capsys):
    iid = _seed(tmp_path)
    code, body, _ = _run(
        tmp_path, ["branch-create", iid, "--revision", "0", *_BRANCH_CREATE], capsys
    )
    assert code == 4
    assert body["status"] == "conflict"
    assert body["reason"] == "revision_mismatch"
    assert body["current_revision"] == 1
    # The conflict is an early return PAST the acquire; the finally still released.
    reg = InvestigationRegistry(tmp_path)
    assert reg.acquire_lease(iid) is True
    reg.release_lease(iid)
    # Document unchanged.
    assert InvestigationRegistry(tmp_path).load(iid)["meta"]["revision"] == 1


# --------------------------------------------------------------------------- #
# Terminal block (exit 3) + declare-unsolved smoke (exit 0, then terminal)      #
# --------------------------------------------------------------------------- #
def test_declare_unsolved_then_terminal_block(tmp_path, capsys):
    iid = _seed(tmp_path)
    c0, b0, _ = _run(
        tmp_path, ["declare-unsolved", iid, "--revision", "1", *_DECLARE_UNSOLVED], capsys
    )
    assert c0 == 0  # DECL-8: declare-unsolved is never gated
    assert b0["terminal_status"] == "unsolved"
    rev = b0["revision"]
    # Any subsequent mutation is terminal-blocked.
    c1, b1, _ = _run(
        tmp_path, ["branch-create", iid, "--revision", str(rev), *_BRANCH_CREATE], capsys
    )
    assert c1 == 3
    assert b1["status"] == "blocked"
    assert b1["reason"] == "investigation_terminal"
    # A read still works.
    c2, b2, _ = _run(tmp_path, ["status", iid], capsys)
    assert c2 == 0 and b2["status"] == "unsolved"


# --------------------------------------------------------------------------- #
# Id containment — BOTH transports (sub-spec §4 / spec §4)                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_id", ["../x", "/abs/path", "z" * 500, "a/b", ".."])
def test_cli_id_containment(tmp_path, capsys, bad_id):
    reg_dir = tmp_path / "reg"
    _seed(reg_dir)
    sentinel = tmp_path / "escaped"  # where reg_dir/../x -like escapes would land
    code, body, _ = _run(reg_dir, ["status", bad_id], capsys)
    assert code == 2
    assert body["status"] == "error"
    assert body["reason"] == "invalid_investigation_id"
    # Nothing was created outside the registry root.
    assert not sentinel.exists()
    assert not (tmp_path / "x").exists()
    assert list(tmp_path.iterdir()) == [reg_dir]


def test_id_containment_shared_on_mcp_path(tmp_path):
    # The SAME malformed id through a SESSION_HELD service (the MCP transport)
    # returns the SAME reason — the hardening is pinned on both transports.
    reg_dir = tmp_path / "reg"
    _seed(reg_dir)
    svc = _mcp_service(reg_dir)
    for bad_id in ("../x", "/abs/path", "z" * 500):
        body = svc.dispatch("investigation_status", {"investigation_id": bad_id})
        assert body == {"status": "error", "reason": "invalid_investigation_id"}
    assert not (tmp_path / "x").exists()
    assert list(tmp_path.iterdir()) == [reg_dir]


def test_id_containment_registry_direct_raises(tmp_path):
    # The shared _dir seam raises for a direct registry caller too (defense in
    # depth), while benign short ids used by the MCP unit suite still pass.
    from mcp_server.registry import InvalidInvestigationId
    reg = InvestigationRegistry(tmp_path)
    for good in ("aaaa0000", "id0", "nope", "does-not-exist"):
        reg.validate_id(good)  # no raise
    for bad in ("../x", "", "..", "a/b", "z" * 500):
        with pytest.raises(InvalidInvestigationId):
            reg.validate_id(bad)


# --------------------------------------------------------------------------- #
# start e2e (--ciphertext, --ciphertext-file incl. stdin, conflicts)           #
# --------------------------------------------------------------------------- #
def test_start_ciphertext_e2e(tmp_path, capsys):
    code, body, _ = _run(
        tmp_path, ["start", "--ciphertext", _CIPHERTEXT, "--language", "en"], capsys
    )
    assert code == 0
    iid = body["investigation_id"]
    # start holds nothing afterward (I-2 §2.3): the lease is free.
    reg = InvestigationRegistry(tmp_path)
    assert reg.acquire_lease(iid) is True
    reg.release_lease(iid)
    # Read it back through the CLI.
    c2, b2, _ = _run(tmp_path, ["status", iid], capsys)
    assert c2 == 0 and b2["investigation_id"] == iid


def test_start_ciphertext_file_stdin(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(icli.sys, "stdin", io.StringIO(_CIPHERTEXT))
    code, body, _ = _run(
        tmp_path, ["start", "--ciphertext-file", "-", "--language", "en"], capsys
    )
    assert code == 0
    assert "investigation_id" in body


def test_start_ciphertext_file_path(tmp_path, capsys):
    ct = tmp_path / "ct.txt"
    ct.write_text(_CIPHERTEXT, encoding="utf-8")
    reg_dir = tmp_path / "reg"
    code, body, _ = _run(reg_dir, ["start", "--ciphertext-file", str(ct)], capsys)
    assert code == 0
    assert "investigation_id" in body


def test_start_ciphertext_inline_and_file_conflict(tmp_path, capsys):
    ct = tmp_path / "ct.txt"
    ct.write_text(_CIPHERTEXT, encoding="utf-8")
    reg_dir = tmp_path / "reg"
    code, body, _ = _run(
        reg_dir,
        ["start", "--ciphertext", _CIPHERTEXT, "--ciphertext-file", str(ct)],
        capsys,
    )
    assert code == 2
    assert body["reason"] == "invalid_cli_arguments"


def test_start_ciphertext_file_and_json_conflict(tmp_path, capsys):
    ct = tmp_path / "ct.txt"
    ct.write_text(_CIPHERTEXT, encoding="utf-8")
    reg_dir = tmp_path / "reg"
    code, body, _ = _run(
        reg_dir,
        ["start", "--ciphertext-file", str(ct),
         "--input-json", json.dumps({"ciphertext": _CIPHERTEXT})],
        capsys,
    )
    assert code == 2
    assert body["reason"] == "invalid_cli_arguments"


def test_start_unreadable_ciphertext_file(tmp_path, capsys):
    reg_dir = tmp_path / "reg"
    missing = tmp_path / "does_not_exist.txt"
    code, body, _ = _run(reg_dir, ["start", "--ciphertext-file", str(missing)], capsys)
    assert code == 2  # file-read failure is a typed CLI input error
    assert body["reason"] == "invalid_cli_arguments"


def test_start_empty_ciphertext_is_domain_result(tmp_path, capsys):
    # Size/format failures remain domain results (exit 1), not CLI input errors.
    code, body, _ = _run(tmp_path, ["start", "--ciphertext", "   "], capsys)
    assert code == 1
    assert body["reason"] == "empty_ciphertext"


# --------------------------------------------------------------------------- #
# Mutation e2e: CLI body == SESSION_HELD dispatch on a copy                     #
# --------------------------------------------------------------------------- #
def test_mutation_e2e_body_parity(tmp_path, capsys):
    reg_a = tmp_path / "a"
    iid = _seed(reg_a)
    reg_b = tmp_path / "b"
    shutil.copytree(reg_a, reg_b)

    # SESSION_HELD oracle mutates copy B.
    oracle = _mcp_service(reg_b)
    oracle_body = oracle.dispatch("hypothesis_branch_create", {
        "investigation_id": iid, "expected_revision": 1,
        "new_name": "h", "cipher_mode": "mono", "rationale": "x",
    })
    oracle.registry.release_lease(iid)

    # CLI mutates copy A with the same input.
    code, cli_body, _ = _run(
        reg_a, ["branch-create", iid, "--revision", "1", *_BRANCH_CREATE], capsys
    )
    assert code == 0
    assert cli_body == oracle_body
    # A follow-up read sees the new revision.
    c2, b2, _ = _run(reg_a, ["status", iid], capsys)
    assert b2["revision"] == 2


def test_call_and_friendly_mutation_parity(tmp_path, capsys):
    # Friendly branch-create and `call hypothesis_branch_create` produce the same
    # service input -> same body (on copies).
    reg_a = tmp_path / "a"
    iid = _seed(reg_a)
    reg_b = tmp_path / "b"
    shutil.copytree(reg_a, reg_b)

    _, friendly_body, _ = _run(
        reg_a, ["branch-create", iid, "--revision", "1", *_BRANCH_CREATE], capsys
    )
    payload = {
        "investigation_id": iid, "expected_revision": 1,
        "new_name": "h", "cipher_mode": "mono", "rationale": "x",
    }
    _, call_body, _ = _run(
        reg_b, ["call", "hypothesis_branch_create", "--input-json", json.dumps(payload)],
        capsys,
    )
    assert call_body == friendly_body


# --------------------------------------------------------------------------- #
# Exclusions (experiment-submit/-collect -> I-3, verify -> I-5)                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("verb,canonical,milestone", [
    ("experiment-submit", "experiment_submit", "I-3"),
    ("experiment-collect", "experiment_collect", "I-3"),
    ("verify", "request_independent_verification", "I-5"),
])
def test_excluded_ops_not_yet_available(tmp_path, capsys, verb, canonical, milestone):
    iid = _seed(tmp_path)
    before = _tree_snapshot(tmp_path)
    # Friendly verb.
    code, body, _ = _run(tmp_path, [verb, iid, "--revision", "1"], capsys)
    assert code == 2
    assert body["status"] == "error"
    assert body["reason"] == "operation_not_yet_available"
    assert milestone in body["detail"]
    # `call` on the canonical name gets the same typed error.
    code2, body2, _ = _run(tmp_path, ["call", canonical, "--input-json", "{}"], capsys)
    assert code2 == 2
    assert body2["reason"] == "operation_not_yet_available"
    assert milestone in body2["detail"]
    # The registry is untouched by either path.
    assert _tree_snapshot(tmp_path) == before


# --------------------------------------------------------------------------- #
# Exit-code matrix — the parent's required classes reachable in I-2             #
# --------------------------------------------------------------------------- #
def test_exit_code_matrix(tmp_path, capsys, monkeypatch):
    iid = _seed(tmp_path)

    # not-found (1)
    c, b, _ = _run(tmp_path, ["status", "ffffffffffff"], capsys)
    assert c == 1 and b["reason"] == "investigation_not_found"

    # schema failure (2): decode missing required branch -> service validation.
    c, b, _ = _run(tmp_path, ["decode", iid], capsys)
    assert c == 2 and b["reason"] == "invalid_arguments"

    # CLI parse failure (2): malformed --input-json.
    c, b, _ = _run(tmp_path, ["status", "--input-json", "{bad"], capsys)
    assert c == 2 and b["reason"] == "invalid_cli_arguments"

    # invalid investigation id (2).
    c, b, _ = _run(tmp_path, ["status", "../x"], capsys)
    assert c == 2 and b["reason"] == "invalid_investigation_id"

    # blocked gate (3): declare-solution with no fresh positive attestation.
    c, b, _ = _run(
        tmp_path,
        ["declare-solution", iid, "--revision", "1", "--branch", "main",
         "--rationale", "done", "--self-confidence", "0.9",
         "--reading-summary", "reads well", "--no-further-iterations-helpful",
         "--further-iterations-note", "n/a"],
        capsys,
    )
    assert c == 3 and b["status"] == "blocked"

    # lease-held (3): a concurrent holder blocks the write.
    other = InvestigationRegistry(tmp_path)
    assert other.acquire_lease(iid) is True
    try:
        c, b, _ = _run(
            tmp_path,
            ["branch-create", iid, "--revision", "2", *_BRANCH_CREATE],
            capsys,
        )
        assert c == 3 and b["reason"] == "writer_lease_held"
    finally:
        other.release_lease(iid)

    # conflict (4): stale revision.
    c, b, _ = _run(
        tmp_path, ["branch-create", iid, "--revision", "999", *_BRANCH_CREATE], capsys
    )
    assert c == 4 and b["reason"] == "revision_mismatch"

    # terminal (3): after a declaration.
    _run(tmp_path, ["declare-unsolved", iid, "--revision", "2", *_DECLARE_UNSOLVED], capsys)
    c, b, _ = _run(
        tmp_path, ["branch-create", iid, "--revision", "3", *_BRANCH_CREATE], capsys
    )
    assert c == 3 and b["reason"] == "investigation_terminal"

    # internal exception (5).
    def _boom(self, name, arguments):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(InvestigationService, "dispatch", _boom)
    c, b, _ = _run(tmp_path, ["status", iid], capsys)
    assert c == 5 and b == {"status": "error", "reason": "internal_error"}
