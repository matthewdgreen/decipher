"""Tests for Workspace + Branch."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace, WorkspaceError


def _build_ws() -> Workspace:
    alpha = Alphabet(list("abc"))
    ct = CipherText(raw="abc ab", alphabet=alpha, separator=" ")
    return Workspace(ct)


class TestBranchLifecycle:
    def test_initial_state(self):
        ws = _build_ws()
        assert ws.branch_names() == ["main"]
        assert ws.get_branch("main").key == {}

    def test_fork_from_main(self):
        ws = _build_ws()
        ws.set_mapping("main", 0, 0)  # a -> A
        ws.fork("exp")
        assert "exp" in ws.branch_names()
        assert ws.get_branch("exp").key == {0: 0}
        assert ws.get_branch("exp").parent == "main"

    def test_fork_isolates_mutations(self):
        ws = _build_ws()
        ws.set_mapping("main", 0, 0)
        ws.fork("exp")
        ws.set_mapping("exp", 1, 1)
        assert ws.get_branch("main").key == {0: 0}
        assert ws.get_branch("exp").key == {0: 0, 1: 1}

    def test_duplicate_fork_errors(self):
        ws = _build_ws()
        ws.fork("exp")
        with pytest.raises(WorkspaceError):
            ws.fork("exp")

    def test_delete_branch(self):
        ws = _build_ws()
        ws.fork("exp")
        ws.delete("exp")
        assert "exp" not in ws.branch_names()

    def test_cannot_delete_main(self):
        ws = _build_ws()
        with pytest.raises(WorkspaceError):
            ws.delete("main")

    def test_invalid_branch_name(self):
        ws = _build_ws()
        with pytest.raises(WorkspaceError):
            ws.fork("bad name with spaces")


class TestDecoding:
    def test_unmapped_shows_question_marks(self):
        ws = _build_ws()
        assert ws.apply_key("main") == "??? ??"

    def test_partial_mapping(self):
        ws = _build_ws()
        ws.set_mapping("main", 0, 0)  # a -> A
        ws.set_mapping("main", 1, 1)  # b -> B
        assert ws.apply_key("main") == "AB? AB"

    def test_full_mapping(self):
        ws = _build_ws()
        ws.set_mapping("main", 0, 0)
        ws.set_mapping("main", 1, 1)
        ws.set_mapping("main", 2, 2)
        assert ws.apply_key("main") == "ABC AB"
        assert ws.is_complete("main")


class TestCompareAndMerge:
    def test_compare(self):
        ws = _build_ws()
        ws.fork("exp")
        ws.set_mapping("main", 0, 0)
        ws.set_mapping("exp", 0, 1)  # disagreement
        ws.set_mapping("exp", 1, 1)  # only in exp
        result = ws.compare("main", "exp")
        assert result["disagreements"] == 1
        assert result["b_only"] == 1
        assert result["agreements"] == 0

    def test_merge_non_conflicting_skips_clashes(self):
        ws = _build_ws()
        ws.fork("exp")
        ws.set_mapping("main", 0, 0)       # main: a->A
        ws.set_mapping("exp", 0, 1)        # exp: a->B (conflict)
        ws.set_mapping("exp", 1, 2)        # exp: b->C (new)
        out = ws.merge("exp", "main", policy="non_conflicting")
        assert out["added"] == 1
        assert out["skipped"] == 1
        # main gained b->C but kept a->A
        assert ws.get_branch("main").key == {0: 0, 1: 2}

    def test_merge_override_wins(self):
        ws = _build_ws()
        ws.fork("exp")
        ws.set_mapping("main", 0, 0)
        ws.set_mapping("exp", 0, 1)
        out = ws.merge("exp", "main", policy="override")
        assert out["overwritten"] == 1
        assert ws.get_branch("main").key == {0: 1}

    def test_merge_non_conflicting_homophony_allowed(self):
        # Homophonic ciphers: two cipher symbols mapping to the same plaintext
        # letter is valid, so non_conflicting merge should NOT skip these.
        ws = _build_ws()
        ws.fork("exp")
        ws.set_mapping("main", 0, 5)   # cipher 0 -> F
        ws.set_mapping("exp", 1, 5)    # cipher 1 -> F (homophone, allowed)
        out = ws.merge("exp", "main", policy="non_conflicting")
        assert out["added"] == 1
        assert out["skipped"] == 0
        assert ws.get_branch("main").key == {0: 5, 1: 5}
