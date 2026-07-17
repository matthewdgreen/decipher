"""MCP registry + revision/lease/terminal semantics (Part 11.2)."""
from __future__ import annotations

import json
import os
import time

import pytest

from mcp_server.registry import (
    InvestigationNotFound,
    InvestigationRegistry,
)
from tests.support.mcp import call, make_server, start


def _meta(**kw):
    base = {
        "investigation_id": "aaaa0000", "label": "", "created_at": 1.0,
        "updated_at": 1.0, "revision": 1, "status": "active", "language": "en",
        "alphabet_size": 5, "tokens": 3, "words": 1, "client_name": "unknown",
        "terminal": None,
    }
    base.update(kw)
    return base


def test_create_load_roundtrip(tmp_path):
    reg = InvestigationRegistry(tmp_path / "reg")
    reg.create(meta=_meta(), state_dict={"hello": "world"})
    doc = reg.load("aaaa0000")
    assert doc["meta"]["revision"] == 1
    assert doc["state"] == {"hello": "world"}
    d = reg.root / "aaaa0000"
    assert (d / "investigation.json").exists()
    assert (d / "meta.json").exists()
    meta_only = json.loads((d / "meta.json").read_text())
    assert meta_only == doc["meta"]


def test_commit_bumps_and_is_atomic(tmp_path):
    reg = InvestigationRegistry(tmp_path / "reg")
    reg.create(meta=_meta(), state_dict={})
    doc = reg.load("aaaa0000")
    before = doc["meta"]["updated_at"]
    time.sleep(0.01)
    new_rev = reg.commit("aaaa0000", doc)
    assert new_rev == 2
    reloaded = reg.load("aaaa0000")
    assert reloaded["meta"]["revision"] == 2
    assert reloaded["meta"]["updated_at"] > before
    # No .tmp residue after atomic rename.
    residue = list((reg.root / "aaaa0000").glob("*.tmp"))
    assert residue == []


def test_list_orders_and_limits(tmp_path):
    reg = InvestigationRegistry(tmp_path / "reg")
    for i in range(3):
        reg.create(meta=_meta(investigation_id=f"id{i}", updated_at=float(i)), state_dict={})
    ordered = reg.list(limit=2)
    assert [m["investigation_id"] for m in ordered] == ["id2", "id1"]


def test_load_unknown_raises(tmp_path):
    reg = InvestigationRegistry(tmp_path / "reg")
    with pytest.raises(InvestigationNotFound):
        reg.load("nope")


def test_unknown_id_typed_result(tmp_path):
    server = make_server(tmp_path)
    body = call(server, "investigation_status", investigation_id="nope")
    assert body["reason"] == "investigation_not_found"


def test_revision_conflict_does_not_mutate(tmp_path):
    server = make_server(tmp_path)
    b = start(server)
    iid = b["investigation_id"]
    # expected_revision one behind (0 vs current 1).
    body = call(server, "hypothesis_branch_create", investigation_id=iid,
                expected_revision=0, new_name="h", cipher_mode="mono", rationale="x")
    assert body["status"] == "conflict"
    assert body["reason"] == "revision_mismatch"
    assert body["current_revision"] == 1
    # On-disk revision unchanged; the branch was NOT created.
    doc = server.registry.load(iid)
    assert doc["meta"]["revision"] == 1


def test_writer_lease_held_across_servers(tmp_path):
    server_a = make_server(tmp_path)
    b = start(server_a)
    iid = b["investigation_id"]
    # server_a holds the lease (acquired at start). A second server on the same
    # registry dir cannot mutate.
    server_b = make_server(tmp_path)
    body = call(server_b, "hypothesis_branch_create", investigation_id=iid,
                expected_revision=1, new_name="h", cipher_mode="mono", rationale="x")
    assert body["status"] == "blocked"
    assert body["reason"] == "writer_lease_held"
    # Release server_a's lease (close the fd), then server_b acquires.
    fd = server_a.registry._leases.pop(iid)
    os.close(fd)
    body = call(server_b, "hypothesis_branch_create", investigation_id=iid,
                expected_revision=1, new_name="h", cipher_mode="mono", rationale="x")
    assert body.get("status") == "ok"


def test_terminal_blocks_mutations_but_reads_work(tmp_path):
    server = make_server(tmp_path)
    b = start(server)
    iid = b["investigation_id"]
    decl = call(server, "meta_declare_unsolved", investigation_id=iid, expected_revision=1,
                rationale="give up", branches_considered=["main"], reading_summary="none",
                further_iterations_helpful=False, further_iterations_note="n/a")
    assert decl["terminal_status"] == "unsolved"
    # A subsequent mutation is blocked; a read still works.
    blocked = call(server, "hypothesis_branch_create", investigation_id=iid,
                   expected_revision=decl["revision"], new_name="h", cipher_mode="m",
                   rationale="x")
    assert blocked["reason"] == "investigation_terminal"
    status = call(server, "investigation_status", investigation_id=iid)
    assert status["status"] == "unsolved"
