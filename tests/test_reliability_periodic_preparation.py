from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import string
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_reliability_periodic.py"
spec = importlib.util.spec_from_file_location("prepare_reliability_periodic", SCRIPT)
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def inputs():
    runtime, rows = [], []
    for i, family in enumerate(("vigenere", "vigenere", "quagmire3", "quagmire3")):
        case_id = "r" + f"{i:012x}"
        runtime.append(packet.runtime_case(case_id, "ABCDEFGHIJKLMNOP", "letters", "en"))
        rows.append({"case_id": case_id, "family": family,
                     "role": "development" if i % 2 == 0 else "held_out",
                     "source_file": f"old_book_{i}", "plaintext": "ANCHOR" + string.ascii_uppercase[i]})
    records = [SimpleNamespace(id=f"p{i:03}", source_file=f"new_book_{i}",
        text=(f"THE {string.ascii_uppercase[i]} JOURNAL DESCRIBES THE RIVER AND THE BOATS " * 30),
        content_hash=f"source_hash_{i}", provenance="fake test corpus") for i in range(14)]
    return records, runtime, {"runtime_manifest_sha256": packet.digest(runtime), "cases": rows}


def test_packet_deterministic_source_disjoint_and_request_firewall():
    records, old_runtime, old = inputs()
    before = json.dumps(old, sort_keys=True)
    runtime, grading, plan = packet.build_packet(records, old_runtime, old)
    assert (runtime, grading, plan) == packet.build_packet(records[::-1], old_runtime, old)
    assert json.dumps(old, sort_keys=True) == before
    assert len(runtime) == len(grading["cases"]) == 16
    fresh = [r for r in grading["cases"] if r["role"] != "regression_anchor"]
    assert len(fresh) == 12
    sources = {r["source_file"] for r in fresh if r["source_file"]}
    assert len(sources) == 11
    assert not sources.intersection(grading["excluded_r0_sources"])
    assert len({r["source_text_sha256"] for r in fresh if r["source_file"]}) == 11
    assert all(set(r) == packet.RUNTIME_FIELDS for r in runtime)
    assert all(r["ciphertext_sha256"] == packet.digest(r["ciphertext"]) for r in runtime)
    assert len(plan["jobs"]) == 32
    assert len({j["job_id"] for j in plan["jobs"]}) == 32
    assert {j["request"]["arm"] for j in plan["jobs"]} == set(packet.ARMS)
    for i in range(0, 32, 2):
        a, b = [j["request"] for j in plan["jobs"][i:i+2]]
        assert a["case_id"] == b["case_id"]
        assert a["arm"] != b["arm"]
        assert all(a[k] == b[k] for k in packet.RUNTIME_FIELDS | {"seed"})
    for job in plan["jobs"]:
        assert set(job["request"]) == packet.REQUEST_FIELDS
        assert job["request_sha256"] == packet.digest(job["request"])
    serialized = json.dumps(plan)
    assert "source_file" not in serialized and "plaintext" not in serialized
    assert not any(r["source_file"] in serialized for r in fresh if r["source_file"])


def test_varied_generation_and_intake_roundtrips():
    runtime, grading, _ = packet.build_packet(*inputs())
    positives = [r for r in grading["cases"] if r["role"] == "fresh_positive"]
    assert [r["period"] for r in positives] == [4, 7, 11, 6, 8, 10]
    assert [r["keyword_length"] for r in positives[3:]] == [6, 7, 8]
    assert all(r["roundtrip_verified"] for r in positives)
    for row in grading["cases"]:
        if row["role"] == "regression_anchor":
            continue
        if row["family"] == "random":
            assert row["plaintext"] is None and row["roundtrip_verified"] is None
            assert row["token_count"] == positives[-1]["token_count"]
        else:
            assert row["roundtrip_verified"]
            assert row["token_count"] >= row["target_letters"]
            assert row["token_count"] == len(row["plaintext"])
            if row["family"] == "columnar_transposition":
                assert len(row["key"]["columnar_keyword"]) == 9
            if row["family"] == "simple_substitution":
                case = next(r for r in runtime if r["case_id"] == row["case_id"])
                assert " " not in case["ciphertext"]


def test_source_exclusion_and_duplicates_cannot_fill_packet():
    records, _, old = inputs()
    prohibited = SimpleNamespace(**vars(records[0]))
    prohibited.id, prohibited.source_file = "aaa", "old_book_0"
    picked = packet.choose_sources([prohibited, *records, *records], old["cases"])
    assert len(picked) == 11 and all(r.source_file != "old_book_0" for r in picked)
    with pytest.raises(ValueError, match="eleven disjoint"):
        packet.choose_sources(records[:10] * 2, old["cases"])
    for record in records:
        record.text = records[0].text
    with pytest.raises(ValueError, match="eleven disjoint"):
        packet.choose_sources(records, old["cases"])


def test_rejects_stale_r0_missing_anchors_and_runtime_leaks():
    records, runtime, old = inputs()
    old["runtime_manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        packet.build_packet(records, runtime, old)
    records, runtime, old = inputs()
    old["cases"].pop()
    with pytest.raises(ValueError, match="four R4"):
        packet.build_packet(records, runtime, old)
    runtime, _, _ = packet.build_packet(*inputs())
    for field in ("plaintext", "key", "family", "period", "role", "source_file"):
        bad = [dict(r) for r in runtime]
        bad[0][field] = "forbidden"
        with pytest.raises(ValueError, match="allowlist"):
            packet.make_plan(bad)
    runtime[0]["ciphertext"] += "A"
    with pytest.raises(ValueError, match="hash"):
        packet.make_plan(runtime)


def test_immutable_freeze_primitive(tmp_path):
    path = tmp_path / "packet.json"
    packet.write_once(path, {"a": 1})
    before = path.read_bytes()
    packet.write_once(path, {"a": 1})
    with pytest.raises(ValueError):
        packet.write_once(path, {"a": 2})
    assert path.read_bytes() == before


def test_preparation_never_calls_search_or_provider(monkeypatch):
    from analysis import polyalphabetic, polyalphabetic_fast
    from automated import runner

    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not solve")

    monkeypatch.setattr(runner, "run_automated", forbidden)
    monkeypatch.setattr(polyalphabetic, "search_periodic_polyalphabetic", forbidden)
    monkeypatch.setattr(polyalphabetic_fast, "search_quagmire3_shotgun_fast", forbidden)
    _, _, plan = packet.build_packet(*inputs())
    assert plan["status"] == "prepared_not_run"
    assert plan["protocol"]["max_runs"] == 32
