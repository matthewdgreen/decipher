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
import contextlib
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import investigation_cli as icli
from agent.model_provider import ModelResponse, ModelUsage, ToolUseBlock
from investigation.experiments import EXPERIMENT_TYPES, register_experiment_type
from investigation.state import InvestigationState
from investigation_service import manifest
from investigation_service.service import InvestigationService, LeasePolicy
from mcp_server.registry import InvestigationRegistry
from tests.support.mcp import apply_basin


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


def _seed_damaged_basin(registry_dir: Path, *, trawler: bool = False) -> tuple[str, int]:
    """Persist a repair-ready keyed basin and return ``(id, revision)``.

    The ordinary BROWN fixture is mechanically acceptable. The TRAWLER fixture
    reproduces the known dictionary/collateral false reject that genuinely
    reaches verifier arbitration.
    """
    plaintext = (
        "THE MISSING TRAWLER RESTED IN THE COVE"
        if trawler else "THE QUICK BROWN FOXES JUMPED"
    )
    service = InvestigationService(
        registry=InvestigationRegistry(registry_dir),
        client_name="test",
        lease_policy=LeasePolicy.SESSION_HELD,
    )
    started = service.dispatch(
        "investigation_start", {"ciphertext": plaintext.lower(), "language": "en"}
    )
    iid = started["investigation_id"]
    runtime = service._runtimes[iid]
    if trawler:
        workspace = runtime.workspace
        cipher_alpha = workspace.cipher_text.alphabet
        plain_alpha = workspace.plaintext_alphabet
        for symbol in cipher_alpha.symbols:
            target = "I" if symbol == "w" else symbol.upper()
            workspace.set_mapping(
                "main", cipher_alpha.id_for(symbol), plain_alpha.id_for(target)
            )
    else:
        apply_basin(runtime)
    service.shutdown()
    service.registry.release_lease(iid)
    revision = InvestigationRegistry(registry_dir).load(iid)["meta"]["revision"]
    return iid, int(revision)


class _PositiveVerifyProvider:
    """One-response local verifier used by CLI transport tests."""

    model = "fake-cli-verify"
    provider_name = "openai"

    def __init__(self) -> None:
        self.calls = 0

    def send(self, **_kwargs):
        self.calls += 1
        verdict = {
            "coherence": 9,
            "reader_accepts": True,
            "reader_accepts_as_solution": True,
            "target_language_confidence": 0.95,
            "semantic_recoverability": 0.90,
            "damage_scope": "local",
            "repairability": "local_repair",
            "uncertainty_note": "",
            "gloss": "reads as clear English",
            "anomalies": [],
            "confidence": "high",
        }
        return ModelResponse(
            content=[
                ToolUseBlock(
                    id="cli_verify_1",
                    name="episode_submit_result",
                    input={"result": verdict, "summary": "reads well"},
                )
            ],
            usage=ModelUsage(50, 10, 0),
        )


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

    # (b) The CLI registers ONE friendly verb per manifest operation (read +
    # create + mutate). The registered set (minus reserved `call`) is exactly
    # the manifest's cli_verb set.
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
    # The private detach-worker verb is registered but reserved (excluded from
    # the parity contract, like `call`).
    assert "_run-experiment" in registered
    all_manifest_verbs = {op.cli_verb for op in manifest.OPERATIONS}
    assert registered - {"call", "_run-experiment"} == all_manifest_verbs
    # The experiment and verification verbs are all public and dispatchable.
    for verb in ("experiment-submit", "experiment-collect", "verify"):
        assert verb in registered


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
    assert "retained_portfolio" in body
    code, body, _ = _run(tmp_path, ["candidate", iid, "--branch", "main"], capsys)
    assert code == 0
    assert body == _direct(tmp_path, "candidate_show", {"investigation_id": iid, "branch": "main"})
    assert body["key_state"]["content_hash"] == body["content_hash"]


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


def test_call_verify_obeys_keyless_policy(tmp_path, capsys):
    iid = _seed(tmp_path)
    before = _tree_snapshot(tmp_path)
    code, body, _ = _run(
        tmp_path,
        [
            "call", "request_independent_verification", "--input-json",
            json.dumps({
                "investigation_id": iid,
                "expected_revision": 1,
                "branch": "main",
            }),
        ],
        capsys,
    )
    assert code == 1
    assert body["status"] == "unavailable"
    assert body["reason"] == "no_verification_provider"
    # The unconditional privacy guard fires before registry/service creation.
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


@pytest.mark.parametrize(
    "policy", [LeasePolicy.INVOCATION_HELD, LeasePolicy.SESSION_HELD],
)
def test_runtime_build_failure_releases_new_lease(tmp_path, monkeypatch, policy):
    """A load/build exception cannot strand a writer lease under either policy."""
    iid = _seed(tmp_path)
    reg = InvestigationRegistry(tmp_path)
    svc = _svc(tmp_path, policy)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("injected runtime construction failure")

    monkeypatch.setattr(svc, "_build_runtime", _boom)
    with pytest.raises(RuntimeError, match="injected runtime construction failure"):
        svc.dispatch("hypothesis_branch_create", {
            "investigation_id": iid, "expected_revision": 1,
            "new_name": "h", "cipher_mode": "mono", "rationale": "x",
        })

    assert svc._runtimes == {}
    assert reg.held_lease_ids() == []
    probe = InvestigationRegistry(tmp_path)
    assert probe.acquire_lease(iid) is True
    probe.release_lease(iid)


@pytest.mark.parametrize(
    "policy", [LeasePolicy.INVOCATION_HELD, LeasePolicy.SESSION_HELD],
)
def test_mutation_reconciles_stale_experiment_with_typed_reason(tmp_path, policy):
    """Writer ownership is the proof used by both transports to orphan stale work."""
    iid = _seed(tmp_path)
    reg = InvestigationRegistry(tmp_path)
    document = reg.load(iid)
    document["state"]["experiment_queue"].append({
        "experiment_id": "exp_stale", "type": "stub_stale",
        "status": "running", "branch": "main", "config": {},
    })
    stale_rev = reg.commit(iid, document)

    svc = _svc(tmp_path, policy)
    body = svc.dispatch("hypothesis_branch_create", {
        "investigation_id": iid, "expected_revision": stale_rev,
        "new_name": "h", "cipher_mode": "mono", "rationale": "x",
    })
    assert body["revision"] == stale_rev + 1
    record = _last_record(tmp_path, iid)
    assert record["status"] == "orphaned"
    assert record["orphan_reason"] == "no_live_worker_at_startup"
    svc.shutdown()


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


# =========================================================================== #
# I-3: experiments — two-commit --wait, --detach handshake, signal cleanup,     #
# startup reconciliation (via the shared loader), cross-surface collect.        #
# =========================================================================== #
_INVOCATION = LeasePolicy.INVOCATION_HELD
_SESSION = LeasePolicy.SESSION_HELD
_TERMINAL = {"completed", "failed", "orphaned"}

_ARBITER_ENV_VARS = (
    "DECIPHER_HOMOPHONIC_PARALLEL_SEEDS",
    "DECIPHER_NULL_MASK_THREADS",
    "DECIPHER_TRANSFORM_RANK_THREADS",
)


@pytest.fixture(autouse=True)
def _arbiter_env_guard():
    """Snapshot + restore the arbiter override vars around every test in this
    module: an in-process async experiment whose worker outlives the service
    shutdown leaves the arbiter env overrides set, which would otherwise leak
    into later modules (e.g. test_experiments)."""
    saved = {v: os.environ.get(v) for v in _ARBITER_ENV_VARS}
    try:
        yield
    finally:
        for v, prior in saved.items():
            if prior is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = prior


def _instant_stub_entry():
    """An EXPERIMENT_TYPES entry whose runner returns instantly (in-process)."""
    def runner(cipher, snapshot, config):
        return {
            "status": "completed", "solver": "stub_instant", "error_message": None,
            "elapsed_seconds": 0.0, "key": {}, "final_decryption": "STUBOK", "steps": [],
        }
    return {
        "config_schema": {"type": "object", "properties": {}},
        "config_defaults": {}, "runner": runner, "description": "instant stub",
    }


def _gated_stub_entry(gate: threading.Event):
    """A runner that blocks on ``gate`` (Events only, no sleeps) — lets a test
    hold an experiment ``running`` while it observes state between the commits."""
    def runner(cipher, snapshot, config):
        gate.wait(timeout=30)
        return {
            "status": "completed", "solver": "stub_gated", "error_message": None,
            "elapsed_seconds": 0.0, "key": {}, "final_decryption": "GATEDOK", "steps": [],
        }
    return {
        "config_schema": {"type": "object", "properties": {}},
        "config_defaults": {}, "runner": runner, "description": "gated stub",
    }


@contextlib.contextmanager
def _registered_type(name: str, entry: dict):
    register_experiment_type(name, entry)
    try:
        yield
    finally:
        EXPERIMENT_TYPES.pop(name, None)


def _svc(registry_dir: Path, policy=_INVOCATION, **kw) -> InvestigationService:
    return InvestigationService(
        registry=InvestigationRegistry(registry_dir), client_name="cli",
        lease_policy=policy, **kw,
    )


def _last_record(registry_dir: Path, iid: str) -> dict | None:
    q = InvestigationRegistry(registry_dir).load(iid)["state"]["experiment_queue"]
    return q[-1] if q else None


def _record_status(registry_dir: Path, iid: str):
    rec = _last_record(registry_dir, iid)
    return rec.get("status") if rec else None


def _wait_until(pred, cap: float = 15.0, interval: float = 0.02) -> bool:
    """Poll ``pred`` until true or ``cap`` seconds elapse (generous hard cap for
    CI robustness; no fixed sleeps in the assertion path)."""
    deadline = time.monotonic() + cap
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return bool(pred())


def _lease_free(registry_dir: Path, iid: str) -> bool:
    reg = InvestigationRegistry(registry_dir)
    got = reg.acquire_lease(iid)
    if got:
        reg.release_lease(iid)
    return got


# --- subprocess stub injection (child processes cannot see monkeypatched types) #
_SITECUSTOMIZE = '''
import time as _t
from investigation.experiments import register_experiment_type

def _instant(cipher, snapshot, config):
    return {"status": "completed", "solver": "stub_instant", "error_message": None,
            "elapsed_seconds": 0.0, "key": {}, "final_decryption": "STUBOK", "steps": []}

def _slow(cipher, snapshot, config):
    _t.sleep(float(config.get("sleep_s") or 120))
    return {"status": "completed", "solver": "stub_slow", "error_message": None,
            "elapsed_seconds": 0.0, "key": {}, "final_decryption": "STUBSLOW", "steps": []}

register_experiment_type("stub_instant", {
    "config_schema": {"type": "object", "properties": {}},
    "config_defaults": {}, "runner": _instant, "description": "instant stub"})
register_experiment_type("stub_slow", {
    "config_schema": {"type": "object", "properties": {"sleep_s": {"type": ["number", "null"]}}},
    "config_defaults": {"sleep_s": 120}, "runner": _slow, "description": "slow stub"})
'''


@pytest.fixture
def stub_dir(tmp_path_factory) -> Path:
    """A directory holding a ``sitecustomize.py`` that registers subprocess stubs;
    prepended to a child's PYTHONPATH so Python auto-imports it at startup."""
    d = tmp_path_factory.mktemp("stubs")
    (d / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    return d


def _child_env(stub_dir: Path) -> dict:
    env = dict(os.environ)
    src = str(Path(icli.__file__).resolve().parent)
    parts = [src, str(stub_dir)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(p for p in parts if p)
    return env


def _spawn_detach_child(registry_dir: Path, obj: dict, stub_dir: Path):
    """Spawn the private ``_run-experiment`` worker directly (mirrors
    ``_run_detached_submit``) but return ``(proc, read_fd)`` so the test controls
    the child and reads its handshake line."""
    read_fd, write_fd = os.pipe()
    os.set_inheritable(write_fd, True)
    argv = [
        sys.executable, "-m", "cli", "investigation",
        "--registry-dir", str(registry_dir), "_run-experiment",
        "--handshake-fd", str(write_fd), "--input-json", json.dumps(obj),
    ]
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True,
        pass_fds=(write_fd,), env=_child_env(stub_dir),
    )
    os.close(write_fd)
    return proc, read_fd


# --------------------------------------------------------------------------- #
# Exclusion narrowing                                                          #
# --------------------------------------------------------------------------- #
def test_experiment_verbs_now_dispatchable(tmp_path, capsys):
    iid = _seed(tmp_path)
    # experiment-collect (no experiment) is an ordinary mutating verb now.
    code, body, _ = _run(tmp_path, ["experiment-collect", iid, "--revision", "1"], capsys)
    assert code == 0
    assert body.get("reason") != "operation_not_yet_available"
    # Verify dispatches too; a keyless CLI gets the I-5 privacy outcome.
    code2, body2, _ = _run(
        tmp_path, ["verify", iid, "--revision", "2", "--branch", "main"], capsys
    )
    assert code2 == 1
    assert body2["status"] == "unavailable"
    assert body2["reason"] == "no_verification_provider"


# --------------------------------------------------------------------------- #
# I-5: explicit verification authority                                        #
# --------------------------------------------------------------------------- #
def test_verify_ignores_ambient_key_without_explicit_provider(
    tmp_path, capsys, monkeypatch,
):
    iid = _seed(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-select-a-provider")
    before = _tree_snapshot(tmp_path)
    code, body, _ = _run(
        tmp_path, ["verify", iid, "--revision", "1", "--branch", "main"], capsys
    )
    assert code == 1
    assert body["reason"] == "no_verification_provider"
    assert _tree_snapshot(tmp_path) == before


def test_verify_requires_allow_external_before_provider_construction(
    tmp_path, capsys, monkeypatch,
):
    iid = _seed(tmp_path)

    def _must_not_construct(*_args, **_kwargs):
        raise AssertionError("provider construction crossed the consent guard")

    monkeypatch.setattr(icli, "_make_cli_verify_provider", _must_not_construct)
    before = _tree_snapshot(tmp_path)
    code, body, _ = _run(
        tmp_path,
        [
            "--verify-provider", "openai",
            "verify", iid, "--revision", "1", "--branch", "main",
        ],
        capsys,
    )
    assert code == 3
    assert body["status"] == "blocked"
    assert body["reason"] == "external_call_not_authorized"
    assert _tree_snapshot(tmp_path) == before


def test_authorized_verify_unlocks_declaration(tmp_path, capsys, monkeypatch):
    iid = _seed(tmp_path)
    provider = _PositiveVerifyProvider()
    monkeypatch.setattr(
        icli, "_make_cli_verify_provider", lambda _name, _model: provider
    )
    code, verified, _ = _run(
        tmp_path,
        [
            "--verify-provider", "openai", "--allow-external",
            "verify", iid, "--revision", "1", "--branch", "main",
        ],
        capsys,
    )
    assert code == 0
    assert verified.get("attestation") is not None
    assert provider.calls == 1

    code2, declared, _ = _run(
        tmp_path,
        [
            "declare-solution", iid,
            "--revision", str(verified["revision"]),
            "--branch", "main",
            "--rationale", "independently verified",
            "--self-confidence", "0.95",
            "--reading-summary", "clear English",
            "--no-further-iterations-helpful",
            "--further-iterations-note", "verification complete",
        ],
        capsys,
    )
    assert code2 == 0
    assert declared["terminal_status"] == "solved"


def test_keyless_declaration_remains_gated(tmp_path, capsys):
    iid = _seed(tmp_path)
    code, body, _ = _run(
        tmp_path,
        [
            "declare-solution", iid, "--revision", "1", "--branch", "main",
            "--rationale", "looks plausible", "--self-confidence", "0.8",
            "--reading-summary", "partial reading",
            "--no-further-iterations-helpful",
            "--further-iterations-note", "none",
        ],
        capsys,
    )
    assert code == 3
    assert body["reason"] == "attestation_required"


def _prepare_cli_repair(registry_dir: Path, capsys, *, trawler: bool = False):
    iid, revision = _seed_damaged_basin(registry_dir, trawler=trawler)
    plaintext = (
        "THE MISSING TRAWLER RESTED IN THE COVE"
        if trawler else "THE QUICK BROWN FOXES JUMPED"
    )
    word = "TRAWLER" if trawler else "BROWN"
    _, reading, _ = _run(
        registry_dir,
        [
            "reading-record", iid, "--revision", str(revision),
            "--branch", "main", "--reading-text", plaintext,
            "--overall-confidence", "0.8",
        ],
        capsys,
    )
    _, compiled, _ = _run(
        registry_dir,
        [
            "repair-test", iid,
            "--revision", str(reading["revision"]),
            "--branch", "main",
            "--hypotheses-json", json.dumps([{"word": word, "word_index": 2}]),
        ],
        capsys,
    )
    assert compiled["status"] == "ok" and compiled["changed_finalists"]
    return iid, compiled, compiled["changed_finalists"][0]["branch"]


def test_mechanical_repair_never_resolves_conditional_provider(
    tmp_path, capsys, monkeypatch,
):
    iid, compiled, winner = _prepare_cli_repair(tmp_path, capsys)

    def _must_not_construct(*_args, **_kwargs):
        raise AssertionError("mechanical acceptance should not resolve a provider")

    monkeypatch.setattr(icli, "_make_cli_verify_provider", _must_not_construct)
    code, body, _ = _run(
        tmp_path,
        [
            "repair-transaction", iid,
            "--revision", str(compiled["revision"]),
            "--branch", "main", "--compile-id", compiled["compile_id"],
            "--winner", winner, "--verifier-arbitration",
        ],
        capsys,
    )
    assert code == 0
    assert body["status"] == "installed"
    assert body["acceptance"]["arbitration"] == {
        "requested": True, "engaged": False,
    }


@pytest.mark.parametrize(
    ("global_args", "expected_code", "expected_reason"),
    [
        ([], 1, "no_verification_provider"),
        (["--verify-provider", "openai"], 3, "external_call_not_authorized"),
    ],
)
def test_needed_arbitration_returns_transport_refusal_without_commit(
    tmp_path, capsys, global_args, expected_code, expected_reason,
):
    iid, compiled, winner = _prepare_cli_repair(tmp_path, capsys, trawler=True)
    before_revision = InvestigationRegistry(tmp_path).load(iid)["meta"]["revision"]
    code, body, _ = _run(
        tmp_path,
        [
            *global_args, "repair-transaction", iid,
            "--revision", str(compiled["revision"]),
            "--branch", "main", "--compile-id", compiled["compile_id"],
            "--winner", winner, "--verifier-arbitration",
        ],
        capsys,
    )
    assert code == expected_code
    assert body["reason"] == expected_reason
    assert body["declaration_gate"] == "closed"
    after_revision = InvestigationRegistry(tmp_path).load(iid)["meta"]["revision"]
    assert after_revision == before_revision


def test_authorized_scripted_arbitration_installs(tmp_path, capsys, monkeypatch):
    iid, compiled, winner = _prepare_cli_repair(tmp_path, capsys, trawler=True)
    provider = _PositiveVerifyProvider()
    monkeypatch.setattr(
        icli, "_make_cli_verify_provider", lambda _name, _model: provider
    )
    code, body, _ = _run(
        tmp_path,
        [
            "--verify-provider", "openai", "--allow-external",
            "repair-transaction", iid,
            "--revision", str(compiled["revision"]),
            "--branch", "main", "--compile-id", compiled["compile_id"],
            "--winner", winner, "--verifier-arbitration",
        ],
        capsys,
    )
    assert code == 0
    assert body["status"] == "installed"
    assert body["acceptance"]["arbitration"]["status"] == "accepted"
    assert provider.calls == 1


# --------------------------------------------------------------------------- #
# Two-commit visibility                                                        #
# --------------------------------------------------------------------------- #
def test_two_commit_visibility(tmp_path):
    iid = _seed(tmp_path)
    gate = threading.Event()
    with _registered_type("stub_gated", _gated_stub_entry(gate)):
        svc = _svc(tmp_path, _INVOCATION)
        obj = {
            "investigation_id": iid, "expected_revision": 1,
            "type": "stub_gated", "branch": "main", "config": {},
        }
        holder: dict = {}

        def _do_run():
            holder["body"], holder["rev"] = svc.run_experiment_to_completion(obj)

        t = threading.Thread(target=_do_run)
        t.start()
        try:
            # COMMIT #1 persists the RUNNING record; a separate registry instance
            # observes it between the two commits.
            assert _wait_until(lambda: _record_status(tmp_path, iid) == "running", cap=10)
            running_rev = InvestigationRegistry(tmp_path).load(iid)["meta"]["revision"]
            assert running_rev == 2
            # A public read has no writer proof and must not reinterpret the
            # other process/thread's live record as an orphan.
            status = _svc(tmp_path, _INVOCATION).dispatch("investigation_status", {
                "investigation_id": iid,
            })
            assert "[running]" in status["brief"]
            assert "[orphaned]" not in status["brief"]
        finally:
            gate.set()  # release the worker so COMMIT #2 can happen
        t.join(timeout=15)
        assert not t.is_alive()

    body = holder["body"]
    assert body.get("experiment_id")
    final_rev = InvestigationRegistry(tmp_path).load(iid)["meta"]["revision"]
    # Two commits: running (2) then terminal (3); the returned body carries the
    # FINAL revision so a collect will not conflict.
    assert final_rev == running_rev + 1
    assert body["revision"] == final_rev
    assert body["status"] == "completed"
    assert body["slots"]["running"] == 0
    assert _record_status(tmp_path, iid) == "completed"

    exp_id = body["experiment_id"]
    collect = _svc(tmp_path, _INVOCATION).dispatch("experiment_collect", {
        "investigation_id": iid, "expected_revision": final_rev,
        "experiment_id": exp_id,
    })
    assert collect.get("status") != "conflict"
    assert collect.get("reason") != "revision_mismatch"


# --------------------------------------------------------------------------- #
# --wait result parity vs an async SESSION_HELD submit                         #
# --------------------------------------------------------------------------- #
def test_wait_result_parity(tmp_path):
    reg_wait = tmp_path / "wait"
    reg_mcp = tmp_path / "mcp"
    iid_w = _seed(reg_wait)
    iid_m = _seed(reg_mcp)
    with _registered_type("stub_instant", _instant_stub_entry()):
        wait_body, wait_rev = _svc(reg_wait, _INVOCATION).run_experiment_to_completion({
            "investigation_id": iid_w, "expected_revision": 1,
            "type": "stub_instant", "branch": "main", "config": {},
        })
        # SESSION_HELD ASYNC submit is the MCP oracle: same domain body, one commit.
        svc_mcp = _svc(reg_mcp, _SESSION, synchronous_experiments=False)
        mcp_body = svc_mcp.dispatch("experiment_submit", {
            "investigation_id": iid_m, "expected_revision": 1,
            "type": "stub_instant", "branch": "main", "config": {},
        })
        svc_mcp.shutdown()

    # R2: --wait reports completion; asynchronous submission reports running.
    # Compare stable submission identity separately from lifecycle fields.
    assert wait_body["revision"] == mcp_body["revision"] + 1
    assert wait_rev == wait_body["revision"]
    a = dict(wait_body)
    b = dict(mcp_body)
    assert a["status"] == "completed" and a["slots"]["running"] == 0
    assert b["status"] == "running" and b["slots"]["running"] == 1
    for d in (a, b):
        d["experiment_id"] = "X"
        d["revision"] = 0
        for lifecycle in ("status", "slots", "summary"):
            d.pop(lifecycle, None)
    assert a == b


# --------------------------------------------------------------------------- #
# SIGINT during --wait (real subprocess)                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.name != "posix", reason="POSIX signals required")
def test_sigint_during_wait(tmp_path, stub_dir):
    reg = tmp_path / "reg"
    iid = _seed(reg)
    argv = [
        sys.executable, "-m", "cli", "investigation",
        "--registry-dir", str(reg), "experiment-submit", iid,
        "--revision", "1", "--type", "stub_slow", "--branch", "main",
        "--config-json", json.dumps({"sleep_s": 120}), "--wait",
    ]
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env=_child_env(stub_dir), text=True,
    )
    try:
        assert _wait_until(lambda: _record_status(reg, iid) == "running", cap=15)
        # Ensure the wait-loop signal handler is installed (microseconds after
        # COMMIT #1); a tiny settle keeps the subprocess test deterministic.
        time.sleep(0.1)
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 130  # conventional SIGINT exit
    # Exactly one JSON object on stdout: the orphan outcome body.
    assert out.endswith("\n")
    body = json.loads(out[:-1])
    assert body["reason"] == "interrupted"
    assert body.get("experiment_id")
    # State finalized: record orphaned/interrupted, lease free.
    rec = _last_record(reg, iid)
    assert rec["status"] == "orphaned"
    assert rec["orphan_reason"] == "interrupted"
    assert _lease_free(reg, iid)


# --------------------------------------------------------------------------- #
# --detach handshake (real subprocess worker)                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.name != "posix", reason="POSIX pipe/session required")
def test_detach_handshake(tmp_path, capsys, monkeypatch, stub_dir):
    reg = tmp_path / "reg"
    iid = _seed(reg)
    # The in-process parent spawns the worker; give the child our stub via env.
    src = str(Path(icli.__file__).resolve().parent)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(p for p in (src, str(stub_dir), os.environ.get("PYTHONPATH", "")) if p),
    )
    code, body, _ = _run(
        reg,
        ["experiment-submit", iid, "--revision", "1", "--type", "stub_instant",
         "--branch", "main", "--detach"],
        capsys,
    )
    assert code == 0
    assert body.get("experiment_id")
    # The detached response carries the SUBMISSION revision (COMMIT #1).
    assert body["revision"] == 2
    # The child completes in the background; the record reaches terminal and the
    # lease frees (the parent never held it).
    assert _wait_until(lambda: _record_status(reg, iid) in _TERMINAL, cap=20)
    assert _record_status(reg, iid) == "completed"
    assert _wait_until(lambda: _lease_free(reg, iid), cap=20)
    # The child committed completion (revision advanced past the submission).
    assert InvestigationRegistry(reg).load(iid)["meta"]["revision"] == 3


def _events(registry_dir: Path, iid: str) -> list[dict]:
    path = registry_dir / iid / "events.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.skipif(os.name != "posix", reason="POSIX pipe required")
def test_detached_worker_crash_reaches_the_event_log(tmp_path, monkeypatch):
    """Sub-spec §3.3: the detached child's stdio is /dev/null, so a crash after
    the handshake must still leave evidence in the investigation event log — and
    the worker must exit non-zero rather than reporting success."""
    iid = _seed(tmp_path)
    handshake = {"status": "running", "experiment_id": "exp_probe", "revision": 2}

    def _boom(self, arguments, *, name="experiment_submit", on_running=None):
        if on_running is not None:
            on_running(handshake)
        raise RuntimeError("injected worker failure")

    monkeypatch.setattr(
        InvestigationService, "run_experiment_to_completion", _boom, raising=True
    )
    read_fd, write_fd = os.pipe()
    args = argparse.Namespace(
        handshake_fd=write_fd, registry_dir=str(tmp_path),
        input_json=json.dumps({
            "investigation_id": iid, "expected_revision": 1,
            "type": "stub_instant", "branch": "main", "config": {},
        }),
    )
    try:
        code = icli._run_experiment_worker(args)
        # The handshake still reached the parent before the crash.
        line, outcome = icli._read_handshake_line(read_fd, 5.0)
    finally:
        os.close(read_fd)
    assert code != 0
    assert outcome == "line"
    assert json.loads(line) == handshake

    events = _events(tmp_path, iid)
    kinds = [e["event"] for e in events]
    assert "detached_worker_started" in kinds
    error = [e for e in events if e["event"] == "detached_worker_error"]
    assert len(error) == 1
    assert "injected worker failure" in error[0]["payload"]["error"]
    assert error[0]["turn"] == 0 and isinstance(error[0]["ts"], float)
    assert "detached_worker_finished" not in kinds


def test_handshake_fd_is_closed_exactly_once(tmp_path, monkeypatch):
    """Review finding #2: the crash test cannot detect a DOUBLE close (a second
    os.close on a dead number raises a swallowed EBADF). Discriminating probe:
    reopen a pipe right after the handshake close so it reuses the just-freed
    descriptor, then crash. If the error path closed the fd a second time it
    would kill one of the reused descriptors."""
    iid = _seed(tmp_path)
    reused: dict[str, int] = {}

    def _boom(self, arguments, *, name="experiment_submit", on_running=None):
        if on_running is not None:
            on_running({"status": "running", "experiment_id": "e", "revision": 2})
        r2, w2 = os.pipe()  # very likely reuses the number just closed
        reused["r"], reused["w"] = r2, w2
        raise RuntimeError("injected after handshake")

    monkeypatch.setattr(
        InvestigationService, "run_experiment_to_completion", _boom, raising=True
    )
    read_fd, write_fd = os.pipe()
    args = argparse.Namespace(
        handshake_fd=write_fd, registry_dir=str(tmp_path),
        input_json=json.dumps({
            "investigation_id": iid, "expected_revision": 1,
            "type": "stub_instant", "branch": "main", "config": {},
        }),
    )
    try:
        code = icli._run_experiment_worker(args)
    finally:
        os.close(read_fd)
    assert code != 0
    # The descriptor number was genuinely recycled, so this is a real probe.
    assert write_fd in (reused["r"], reused["w"])
    try:
        os.fstat(reused["r"])  # would raise EBADF if double-closed
        os.fstat(reused["w"])
    finally:
        for fd in (reused["r"], reused["w"]):
            try:
                os.close(fd)
            except OSError:
                pass


def test_detach_handshake_timeout_requires_a_live_child(tmp_path, capsys, monkeypatch):
    """Review finding #3: F2 changed the timeout branch's precondition (the
    child must still be ALIVE), and nothing pinned it. A child that never writes
    and outlives the deadline is a genuine timeout, not an exit."""
    iid = _seed(tmp_path)
    sleeper = tmp_path / "sleeper.sh"
    sleeper.write_text("#!/bin/sh\nsleep 30\n")
    sleeper.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(sleeper))
    monkeypatch.setattr(icli, "_DETACH_HANDSHAKE_TIMEOUT_S", 1.0)

    started = time.monotonic()
    code, body, _ = _run(
        tmp_path,
        ["experiment-submit", iid, "--revision", "1", "--type", "stub_instant",
         "--branch", "main", "--config-json", "{}", "--detach"],
        capsys,
    )
    elapsed = time.monotonic() - started
    assert code == 5
    assert body["reason"] == "detach_handshake_timeout", body
    assert elapsed < 10.0, f"took {elapsed:.1f}s; the 1s deadline did not apply"
@pytest.mark.skipif(
    os.name != "posix" or not os.path.exists("/usr/bin/false"),
    reason="POSIX /usr/bin/false required",
)
def test_detach_worker_exit_before_handshake_is_not_reported_as_timeout(
    tmp_path, capsys, monkeypatch,
):
    """A child that dies in milliseconds closes the pipe (EOF), which reads the
    same as a timeout. The parent must distinguish them from the child's exit
    status instead of claiming the worker 'may still be running'."""
    iid = _seed(tmp_path)
    # Point the spawn at a binary that exits immediately without a handshake.
    monkeypatch.setattr(sys, "executable", "/usr/bin/false")
    started = time.monotonic()
    code, body, _ = _run(
        tmp_path,
        ["experiment-submit", iid, "--revision", "1", "--type", "stub_instant",
         "--branch", "main", "--detach"],
        capsys,
    )
    assert code == 5
    assert body["reason"] == "detach_worker_exited"
    assert "exited with status" in body["detail"]
    # It returned on the child's death, not after the 60s handshake timeout.
    assert time.monotonic() - started < 30.0
    # Nothing was submitted and no lease leaked.
    assert _record_status(tmp_path, iid) is None
    assert _lease_free(tmp_path, iid)


# --------------------------------------------------------------------------- #
# SIGKILL reconciliation (real subprocess worker)                              #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.name != "posix", reason="POSIX signals/session required")
def test_kill_reconciliation(tmp_path, stub_dir):
    reg = tmp_path / "reg"
    iid = _seed(reg)
    obj = {
        "investigation_id": iid, "expected_revision": 1,
        "type": "stub_slow", "branch": "main", "config": {"sleep_s": 120},
    }
    proc, read_fd = _spawn_detach_child(reg, obj, stub_dir)
    try:
        line, _outcome = icli._read_handshake_line(read_fd, 20.0)
        assert line is not None, "child never sent a submission handshake"
        submit_body = json.loads(line)
        exp_id = submit_body["experiment_id"]
        assert submit_body["status"] == "running"
        submit_rev = submit_body["revision"]
        # The CHILD holds the lease (the parent/test never did).
        probe = InvestigationRegistry(reg)
        assert probe.acquire_lease(iid) is False
        assert _record_status(reg, iid) == "running"
    finally:
        os.close(read_fd)

    proc.kill()  # SIGKILL — cannot orphan gracefully
    proc.wait(timeout=15)
    # flock died with the process: the lease is free again.
    assert _wait_until(lambda: _lease_free(reg, iid), cap=15)
    # The record is still persisted as running (COMMIT #1 only).
    assert _record_status(reg, iid) == "running"

    # A read has no writer proof, so it honestly reports the persisted state.
    status = _svc(reg, _INVOCATION).dispatch("investigation_status", {
        "investigation_id": iid,
    })
    assert "[running]" in status["brief"]

    # The NEXT CLI mutation acquires the free writer lease, which proves the old
    # worker is gone, then reconciles + persists the orphaned record.
    svc_cli = _svc(reg, _INVOCATION)
    m = svc_cli.dispatch("hypothesis_branch_create", {
        "investigation_id": iid, "expected_revision": submit_rev,
        "new_name": "h", "cipher_mode": "mono", "rationale": "x",
    })
    assert m["revision"] == submit_rev + 1
    rec = _last_record(reg, iid)
    assert rec["status"] == "orphaned"
    assert rec["orphan_reason"] == "no_live_worker_at_startup"

    # resubmit re-runs from the stored snapshot (async SESSION_HELD so it returns
    # immediately). The stub type must be registered IN THIS process too (the
    # `type` field is schema-required and validated here); the in-process slow
    # stub blocks harmlessly on a never-set gate (daemon thread, killed at exit).
    never = threading.Event()

    def _blocked(cipher, snapshot, config):
        never.wait(timeout=2)
        return {"status": "completed", "solver": "stub_slow", "final_decryption": ""}

    slow_entry = {
        "config_schema": {"type": "object", "properties": {"sleep_s": {"type": ["number", "null"]}}},
        "config_defaults": {"sleep_s": 120}, "runner": _blocked, "description": "slow",
    }
    with _registered_type("stub_slow", slow_entry):
        svc_async = _svc(reg, _SESSION, synchronous_experiments=False)
        cur = InvestigationRegistry(reg).load(iid)["meta"]["revision"]
        rr = svc_async.dispatch("experiment_submit", {
            "investigation_id": iid, "expected_revision": cur,
            "type": "stub_slow", "resubmit": exp_id,
        })
        assert rr.get("resubmitted_from") == exp_id
        assert rr["experiment_id"] != exp_id
        # The duplicate-spec dedup NEVER returns the orphaned record.
        cur = rr["revision"]
        ds = svc_async.dispatch("experiment_submit", {
            "investigation_id": iid, "expected_revision": cur,
            "type": "stub_slow", "branch": "main", "config": {"sleep_s": 120},
        })
        assert ds["experiment_id"] != exp_id
        assert not ds.get("deduplicated")
        svc_async.shutdown()


# --------------------------------------------------------------------------- #
# Cross-surface collect (both directions)                                      #
# --------------------------------------------------------------------------- #
def test_cross_surface_cli_submit_mcp_collect(tmp_path):
    iid = _seed(tmp_path)
    with _registered_type("stub_instant", _instant_stub_entry()):
        body, final_rev = _svc(tmp_path, _INVOCATION).run_experiment_to_completion({
            "investigation_id": iid, "expected_revision": 1,
            "type": "stub_instant", "branch": "main", "config": {},
        })
        exp_id = body["experiment_id"]
        assert body["revision"] == final_rev
        svc_mcp = _svc(tmp_path, _SESSION)
        packet = svc_mcp.dispatch("experiment_collect", {
            "investigation_id": iid, "expected_revision": final_rev,
            "experiment_id": exp_id, "install": True,
        })
        svc_mcp.registry.release_lease(iid)
    assert packet.get("status") == "completed"
    assert packet.get("installed_as")


def test_cross_surface_mcp_submit_cli_collect(tmp_path):
    iid = _seed(tmp_path)
    with _registered_type("stub_instant", _instant_stub_entry()):
        svc_mcp = _svc(tmp_path, _SESSION, synchronous_experiments=True)
        sub = svc_mcp.dispatch("experiment_submit", {
            "investigation_id": iid, "expected_revision": 1,
            "type": "stub_instant", "branch": "main", "config": {},
        })
        exp_id = sub["experiment_id"]
        svc_mcp.registry.release_lease(iid)  # release WITHOUT the shutdown re-commit
        rev = InvestigationRegistry(tmp_path).load(iid)["meta"]["revision"]
        packet = _svc(tmp_path, _INVOCATION).dispatch("experiment_collect", {
            "investigation_id": iid, "expected_revision": rev,
            "experiment_id": exp_id, "install": True,
        })
    assert packet.get("status") == "completed"
    assert packet.get("installed_as")
