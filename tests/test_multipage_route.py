"""Tests for the Phase-2.4 multipage shared-key benchmark route.

Covers the group-definition loader + validation, the shared symbol-inventory
policy, the refinement split, the two-page end-to-end route with a stubbed
homophonic solve (offsets/projection round-trip, per-page + group artifact
shapes, group metadata), the group-native word-repair adoption, and the
lazy-import binding constraint extended to ``automated.multipage_route``. The
heavy homophonic solve and the ``propose_word_repairs`` menu are stubbed so
these tests stay fast and deterministic.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import automated.runner as runner
from automated import multipage_route
from automated.multipage_route import (
    load_group_definition,
    run_automated_multipage,
    validate_shared_symbol_inventory,
    _split_refinement,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeLoader:
    """Minimal BenchmarkLoader stand-in.

    ``data`` maps test_id -> (canonical_transcription, plaintext). ``load_tests``
    ignores the split and returns a BenchmarkTest-like object per id.
    """

    def __init__(self, data: dict[str, tuple[str, str]]) -> None:
        self._data = data

    def load_tests(self, split):  # noqa: ARG002 - split unused in the fake
        return [SimpleNamespace(test_id=tid) for tid in self._data]

    def load_test_data(self, test):
        canonical, plaintext = self._data[test.test_id]
        return SimpleNamespace(canonical_transcription=canonical, plaintext=plaintext)


def _canned_identity_homophonic(cipher_text, language, **_kwargs):
    """Decode each cipher symbol to its own (id mod 26) letter.

    Reproduces a letter-per-token stand-in for the ciphertext (never the
    plaintext), so the 'solve' is ground-truth-free and deterministic.
    """
    key = {tid: (tid % 26) for tid in range(cipher_text.alphabet.size)}
    decryption = "".join(chr(ord("A") + key[t]) for t in cipher_text.tokens)
    step = {"name": "search_homophonic_anneal", "solver": "native_homophonic_anneal"}
    return "native_homophonic_anneal", key, decryption, step


@pytest.fixture
def stub_homophonic(monkeypatch):
    monkeypatch.setattr(runner, "_run_homophonic", _canned_identity_homophonic)


@pytest.fixture
def stub_model(monkeypatch):
    """Skip real binary-model resolution in unit tests (word-repair path)."""
    monkeypatch.setattr(runner, "zenith_native_model_path", lambda _lang: None)


def _two_page_data() -> dict[str, tuple[str, str]]:
    # Pages share S002/S003; A also has S001/S004, B also S005. Every token id
    # is < 26 so the identity key maps each to a distinct letter (invertible).
    return {
        "pgA": ("S001 S002 S003 S001 S004", "ABCAD"),
        "pgB": ("S002 S003 S005", "BCE"),
    }


def _group(test_ids, name="tg", language="en"):
    return {
        "name": name,
        "benchmark_split": "x.jsonl",
        "language": language,
        "description": "test group",
        "test_ids": list(test_ids),
    }


# ---------------------------------------------------------------------------
# Group definition loader
# ---------------------------------------------------------------------------


def _write_group(tmp_path, payload) -> str:
    path = tmp_path / "group.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_load_group_definition_valid(tmp_path):
    payload = _group(["a", "b"], name="grp", language="de")
    data = load_group_definition(_write_group(tmp_path, payload))
    assert data["name"] == "grp"
    assert data["test_ids"] == ["a", "b"]
    assert data["language"] == "de"


def test_load_group_definition_ships_copiale_evidence():
    """The shipped group has the five evidence-packet Track-B test ids."""
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    path = os.path.join(repo_root, "frontier", "groups", "copiale_evidence.json")
    data = load_group_definition(path)
    assert data["name"] == "copiale_evidence"
    assert data["benchmark_split"] == "copiale_tests.jsonl"
    assert data["language"] == "de"
    assert data["test_ids"] == [
        "copiale_single_B_copiale_p017",
        "copiale_single_B_copiale_p035",
        "copiale_single_B_copiale_p052",
        "copiale_single_B_copiale_p068",
        "copiale_single_B_copiale_p084",
    ]

    # And those ids are exactly the copiale_evidence rows in the evidence packet.
    packet_path = os.path.join(repo_root, "frontier", "copiale_evidence_packet.jsonl")
    packet_ids = []
    with open(packet_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "copiale_evidence" in (row.get("frontier_tags") or []):
                packet_ids.append(row["test_id"])
    assert sorted(data["test_ids"]) == sorted(packet_ids)


def test_load_group_definition_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_group_definition(str(tmp_path / "nope.json"))


def test_load_group_definition_bad_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_group_definition(str(path))


@pytest.mark.parametrize("drop", ["name", "benchmark_split", "test_ids", "language"])
def test_load_group_definition_missing_required_key(tmp_path, drop):
    payload = _group(["a", "b"])
    payload.pop(drop)
    with pytest.raises(ValueError, match="missing required key"):
        load_group_definition(_write_group(tmp_path, payload))


def test_load_group_definition_empty_test_ids(tmp_path):
    payload = _group([])
    with pytest.raises(ValueError, match="non-empty list"):
        load_group_definition(_write_group(tmp_path, payload))


def test_load_group_definition_duplicate_test_ids(tmp_path):
    payload = _group(["a", "a"])
    with pytest.raises(ValueError, match="duplicates"):
        load_group_definition(_write_group(tmp_path, payload))


@pytest.mark.parametrize("bad_name", ["../x", "a/b", "a\\b", "a.b", "..", ".hidden"])
def test_load_group_definition_rejects_path_unsafe_name(tmp_path, bad_name):
    """F5: the group name becomes an artifact directory component
    (<artifact_dir>/automated_multipage/<name>/), so separators and dots are
    rejected at load time."""
    payload = _group(["a", "b"], name=bad_name)
    with pytest.raises(ValueError, match="path-safe slug"):
        load_group_definition(_write_group(tmp_path, payload))


# ---------------------------------------------------------------------------
# Shared symbol-inventory policy
# ---------------------------------------------------------------------------


def test_validate_shared_symbol_inventory_accepts_s_tokens():
    loader = _FakeLoader(_two_page_data())
    tests = {t.test_id: t for t in loader.load_tests("x")}
    # Does not raise.
    validate_shared_symbol_inventory(loader, tests, ["pgA", "pgB"])


def test_validate_shared_symbol_inventory_rejects_non_s_token():
    """A page with a raw-letter transcription is rejected before combining."""
    data = {
        "pgA": ("S001 S002 S003", "ABC"),
        "pgB": ("A B C D", "ABCD"),  # not S-tokens
    }
    loader = _FakeLoader(data)
    tests = {t.test_id: t for t in loader.load_tests("x")}
    with pytest.raises(ValueError, match="symbol-inventory policy"):
        validate_shared_symbol_inventory(loader, tests, ["pgA", "pgB"])


def test_validate_shared_symbol_inventory_rejects_empty_page():
    data = {"pgA": ("S001 S002", "AB"), "pgB": ("", "")}
    loader = _FakeLoader(data)
    tests = {t.test_id: t for t in loader.load_tests("x")}
    with pytest.raises(ValueError, match="empty transcription"):
        validate_shared_symbol_inventory(loader, tests, ["pgA", "pgB"])


def test_run_automated_multipage_rejects_mismatched_group(stub_homophonic, tmp_path):
    data = {
        "pgA": ("S001 S002 S003", "ABC"),
        "pgB": ("XY ZZ", "PQ"),  # non-S-token page
    }
    loader = _FakeLoader(data)
    with pytest.raises(ValueError, match="symbol-inventory policy"):
        run_automated_multipage(
            loader=loader,
            group=_group(["pgA", "pgB"]),
            homophonic_budget="screen",
            homophonic_refinement="none",
            artifact_dir=str(tmp_path),
        )


def test_run_automated_multipage_missing_test_id_raises(stub_homophonic, tmp_path):
    loader = _FakeLoader(_two_page_data())
    with pytest.raises(ValueError, match="not found in split"):
        run_automated_multipage(
            loader=loader,
            group=_group(["pgA", "missing"]),
            homophonic_budget="screen",
            homophonic_refinement="none",
            artifact_dir=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# Refinement split
# ---------------------------------------------------------------------------


def test_split_refinement_variants():
    assert _split_refinement("none") == ("none", False)
    assert _split_refinement("null_masks") == ("null_masks", False)
    assert _split_refinement("word_repair") == ("none", True)
    assert _split_refinement("null_masks+word_repair") == ("null_masks", True)
    # A non-word-repair refinement passes through untouched.
    assert _split_refinement("two_stage") == ("two_stage", False)


# ---------------------------------------------------------------------------
# End-to-end route (stubbed solve)
# ---------------------------------------------------------------------------


def test_run_automated_multipage_two_page_end_to_end(stub_homophonic, tmp_path):
    loader = _FakeLoader(_two_page_data())
    result = run_automated_multipage(
        loader=loader,
        group=_group(["pgA", "pgB"], name="grp2"),
        homophonic_budget="screen",
        homophonic_refinement="none",
        artifact_dir=str(tmp_path),
    )

    assert result.status == "completed"
    assert result.group_name == "grp2"
    assert result.test_ids == ["pgA", "pgB"]
    # Combined cipher = 5 + 3 tokens; shared alphabet has 5 distinct symbols.
    assert result.combined_token_count == 8
    assert result.combined_alphabet_size == 5
    assert len(result.combined_cipher_sha256) == 64
    assert result.selected_mask == []

    # Identity key projects each page's tokens to distinct letters -> exact match
    # against the plaintext we supplied, i.e. round-trip through build+project.
    by_id = {pr.test_id: pr for pr in result.page_results}
    assert by_id["pgA"].decryption == "ABCAD"
    assert by_id["pgB"].decryption == "BCE"
    assert by_id["pgA"].char_accuracy == 1.0
    assert by_id["pgB"].char_accuracy == 1.0
    assert result.aggregate_char_accuracy == 1.0

    # Group artifact: combined solve steps + per-page projections + aggregate.
    group_artifact = json.loads(open(result.group_artifact_path, encoding="utf-8").read())
    assert group_artifact["run_mode"] == "automated_multipage"
    assert group_artifact["group_name"] == "grp2"
    assert group_artifact["combined"]["cipher_sha256"] == result.combined_cipher_sha256
    step_names = [s.get("name") for s in group_artifact["combined_solve_steps"]]
    assert "route_automated_solver" in step_names
    assert "search_homophonic_anneal" in step_names
    assert [p["test_id"] for p in group_artifact["pages"]] == ["pgA", "pgB"]
    # Per-page token offsets in the combined stream.
    assert [p["token_offset"] for p in group_artifact["pages"]] == [0, 5]
    assert group_artifact["aggregate"]["page_count"] == 2

    # Per-page artifact: single-page automated schema + multipage_group metadata.
    page_dir = os.path.dirname(by_id["pgA"].artifact_path)
    assert os.path.basename(by_id["pgA"].artifact_path) == "pgA.json"
    pa = json.loads(open(by_id["pgA"].artifact_path, encoding="utf-8").read())
    assert pa["run_mode"] == "automated_multipage"
    assert pa["test_id"] == "pgA"
    assert pa["decryption"] == "ABCAD"
    assert pa["multipage_group"]["name"] == "grp2"
    assert pa["multipage_group"]["page_index"] == 0
    assert pa["multipage_group"]["page_count"] == 2
    assert pa["multipage_group"]["token_offset"] == 0
    assert pa["multipage_group"]["combined_cipher_sha256"] == result.combined_cipher_sha256
    # The shared key is identical across page artifacts (one shared key), and
    # the group artifact records the same FINAL shared key at top level (F4).
    pb = json.loads(open(by_id["pgB"].artifact_path, encoding="utf-8").read())
    assert pa["key"] == pb["key"]
    assert group_artifact["key"] == pa["key"]
    assert pb["multipage_group"]["token_offset"] == 5
    # Both page artifacts live under the same run directory (named by run_id).
    assert os.path.basename(page_dir) == result.run_id
    assert os.path.dirname(by_id["pgB"].artifact_path) == page_dir


def test_run_automated_multipage_word_repair_runs_on_full_group(
    stub_homophonic, stub_model, monkeypatch, tmp_path
):
    """Word repair is invoked with the REAL page group (not a combined single
    page), and an adopted edit rewrites the shared key + re-projects each page."""
    import analysis.multipage as multipage
    import analysis.word_hypothesis_repair as whr
    from analysis.candidate_packet import CandidatePacket

    # Baseline validation stub (as the runner word-repair tests do).
    monkeypatch.setattr(
        multipage,
        "score_page_runtime",
        lambda row, *, key, mask, language, model_path=None: {
            "validation_score_v2": 1.0,
            "validation_components_v2": {},
            "language_quality_mean": 0.0,
            "language_quality_features": {},
            "dict_rate": 0.0,
            "diagnostics": {},
            "test_id": row["test_id"],
        },
    )

    captured = {}

    def fake_propose(**kwargs):
        # Prove the group is passed through: capture the pages list.
        captured["pages"] = kwargs["pages"]
        captured["shared_key"] = dict(kwargs["shared_key"])
        return [
            CandidatePacket(
                candidate_id="edits:S001:A->X",
                kind="word_repair",
                source={"solver": "word_hypothesis_repair"},
                rank=1,
                text=None,
                preview="X..",
                solver_scores={"page_validation_avg": 2.5, "adjudication_score": 5.0},
                validation={"accepted": True, "decision": "runtime_accept", "reasons": []},
                provenance={"edits": ["S001:A->X"], "mask": []},
            )
        ]

    monkeypatch.setattr(whr, "propose_word_repairs", fake_propose)
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_ADOPT", "1")

    loader = _FakeLoader(_two_page_data())
    result = run_automated_multipage(
        loader=loader,
        group=_group(["pgA", "pgB"]),
        homophonic_budget="screen",
        homophonic_refinement="word_repair",
        artifact_dir=str(tmp_path),
    )

    # propose_word_repairs saw the full 2-page group (cross-page evidence case),
    # with plaintext stripped by construction (F1 firewall).
    assert [p.test_id for p in captured["pages"]] == ["pgA", "pgB"]
    assert all(p.plaintext == "" for p in captured["pages"])

    # The adopted edit reassigned S001 (token id 0) A->X on the shared key, so
    # every page that uses S001 now decodes it as X.
    assert result.word_repair is not None
    assert result.word_repair["adopted"] is not None
    by_id = {pr.test_id: pr for pr in result.page_results}
    assert by_id["pgA"].decryption == "XBCXD"  # S001->X in positions 0 and 3
    assert by_id["pgB"].decryption == "BCE"  # pgB never uses S001

    # The word-repair step is recorded on the group artifact, not per page.
    group_artifact = json.loads(open(result.group_artifact_path, encoding="utf-8").read())
    assert group_artifact["word_repair"]["name"] == "search_word_repair"
    assert group_artifact["word_repair"]["adopted"]["edits"] == ["S001:A->X"]
    # F4: the group artifact's top-level key is the FINAL post-adoption shared
    # key (S001 = token 0 -> X = 23), while combined.key keeps the pre-repair
    # solve key (token 0 -> A = 0).
    assert group_artifact["key"]["0"] == ord("X") - ord("A")
    assert group_artifact["combined"]["key"]["0"] == 0


def test_run_automated_multipage_word_repair_menu_only_default(
    stub_homophonic, stub_model, monkeypatch, tmp_path
):
    """With DECIPHER_WORD_REPAIR_ADOPT unset, the group gate is measurement-only:
    would_adopt is recorded but the shared key/projection is unchanged."""
    import analysis.multipage as multipage
    import analysis.word_hypothesis_repair as whr
    from analysis.candidate_packet import CandidatePacket

    monkeypatch.delenv("DECIPHER_WORD_REPAIR_ADOPT", raising=False)
    monkeypatch.setattr(
        multipage,
        "score_page_runtime",
        lambda row, *, key, mask, language, model_path=None: {
            "validation_score_v2": 1.0,
            "validation_components_v2": {},
            "language_quality_mean": 0.0,
            "language_quality_features": {},
            "dict_rate": 0.0,
            "diagnostics": {},
            "test_id": row["test_id"],
        },
    )
    monkeypatch.setattr(
        whr,
        "propose_word_repairs",
        lambda **_kwargs: [
            CandidatePacket(
                candidate_id="edits:S001:A->X",
                kind="word_repair",
                source={},
                rank=1,
                text=None,
                preview="X..",
                solver_scores={"page_validation_avg": 2.5, "adjudication_score": 5.0},
                validation={"accepted": True, "decision": "runtime_accept", "reasons": []},
                provenance={"edits": ["S001:A->X"], "mask": []},
            )
        ],
    )

    loader = _FakeLoader(_two_page_data())
    result = run_automated_multipage(
        loader=loader,
        group=_group(["pgA", "pgB"]),
        homophonic_budget="screen",
        homophonic_refinement="word_repair",
        artifact_dir=str(tmp_path),
    )

    # Menu-only: the projection is the unmodified identity projection.
    by_id = {pr.test_id: pr for pr in result.page_results}
    assert by_id["pgA"].decryption == "ABCAD"
    assert result.word_repair_adopt_enabled is False
    # The gate still records what it WOULD adopt.
    assert result.word_repair["would_adopt"] is not None
    assert result.word_repair["adopted"] is None


# ---------------------------------------------------------------------------
# Lazy-import binding constraint (extended to the new module)
# ---------------------------------------------------------------------------


def test_multipage_route_does_not_import_promoted_libraries_at_module_level():
    """Binding constraint (Phase 2.4): analysis.multipage /
    analysis.word_hypothesis_repair must be lazily imported in
    automated.multipage_route too, since analysis.multipage imports
    automated.runner at its top."""
    source_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "automated", "multipage_route.py"
    )
    with open(source_path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    banned = {"analysis.multipage", "analysis.word_hypothesis_repair"}
    for node in tree.body:  # module-level statements only
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in banned, f"top-level import of {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in banned, f"top-level from-import of {node.module}"

    src_root = os.path.join(os.path.dirname(__file__), "..", "src")
    probe = textwrap.dedent(
        """
        import sys
        import automated.multipage_route  # noqa: F401
        assert "analysis.multipage" not in sys.modules, "multipage imported at load"
        assert "analysis.word_hypothesis_repair" not in sys.modules, "whr imported at load"
        print("ok")
        """
    )
    env = dict(os.environ, PYTHONPATH=src_root)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
