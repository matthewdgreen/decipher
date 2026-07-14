"""A2 declaration-policy tests.

The exhaustive v2 declare-gate behavior lives in test_agent_reliability.py
(including the two dedicated pin tests). Here we cover the injected-policy
surface itself: the executor default is V2GatePolicy (byte-identical gates),
and NoGatesPolicy bypasses BOTH declare tools.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.tools_v2 import (
    DeclarationPolicy,
    NoGatesPolicy,
    V2GatePolicy,
    WorkspaceToolExecutor,
)
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


def _executor(raw: str, *, separator: str | None = None, policy: DeclarationPolicy | None = None):
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=separator)
    return WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "DOG"},
        word_list=["THE", "DOG"],
        pattern_dict={},
        declaration_policy=policy,
    )


def test_executor_default_policy_is_v2_gates():
    ex = _executor("ABC")
    assert isinstance(ex._declaration_policy, V2GatePolicy)


def test_no_gates_policy_bypasses_branch_cards_gate():
    # Multiple branches + no branch_cards call → V2 gate would block.
    ex = _executor("ABC", policy=NoGatesPolicy())
    ex.workspace.fork("candidate", from_branch="main")

    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Declaring directly under NoGatesPolicy.",
        "self_confidence": 0.5,
    })

    assert out["status"] == "ok"
    assert out["accepted"] is True
    assert ex.terminated is True
    assert ex.solution is not None
    assert ex.solution.branch == "main"


def test_v2_policy_blocks_same_scenario_no_gates_allows():
    # Control: the same scenario is blocked under the default v2 policy.
    ex_v2 = _executor("ABC")
    ex_v2.workspace.fork("candidate", from_branch="main")
    blocked = ex_v2._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Declaring directly.",
        "self_confidence": 0.5,
    })
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "branch_cards_required"
    assert ex_v2.terminated is False


def test_no_gates_policy_bypasses_reading_workflow_gate():
    ex = _executor("AP PLY", separator=" ", policy=NoGatesPolicy())
    ex.set_max_iterations(10)
    ex.set_iteration(4)
    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Readable text remains, but there are boundary issues.",
        "self_confidence": 0.7,
    })
    assert out["status"] == "ok"
    assert out["accepted"] is True
    assert ex.terminated is True


def test_no_gates_policy_bypasses_declare_unsolved_gates():
    # more-work-requested gate would block under v2; NoGatesPolicy allows.
    ex = _executor("ABC", policy=NoGatesPolicy())
    ex.set_max_iterations(20)
    ex.set_iteration(3)
    out = ex._tool_meta_declare_unsolved({
        "rationale": "No branch reads as language.",
        "further_iterations_helpful": True,
        "best_branch": "main",
    })
    assert out["status"] == "ok"
    assert out["accepted"] is True
    assert out["outcome"] == "unsolved"
    assert ex.terminated is True
    assert ex.unsolved_declaration is not None


def test_v2_policy_blocks_unsolved_more_work_no_gates_allows():
    ex = _executor("ABC")
    ex.set_max_iterations(20)
    ex.set_iteration(3)
    blocked = ex._tool_meta_declare_unsolved({
        "rationale": "No branch reads as language.",
        "further_iterations_helpful": True,
        "best_branch": "main",
    })
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "unsolved_but_more_work_requested"
    assert ex.terminated is False


def test_no_gates_policy_still_validates_missing_branch():
    # NoGatesPolicy bypasses gates but the handler's validity check remains.
    ex = _executor("ABC", policy=NoGatesPolicy())
    out = ex._tool_meta_declare_solution({
        "branch": "does_not_exist",
        "rationale": "x",
        "self_confidence": 0.5,
    })
    assert "error" in out
    assert ex.terminated is False
