"""Workspace abstraction for the v2 agent.

A Workspace holds the cipher (immutable) and a set of named branches, each
with an independent partial key. Tools take explicit branch arguments; there
is no implicit "current branch".
"""
from workspace.branch import Branch
from workspace.workspace import Workspace, WorkspaceError

__all__ = ["Branch", "Workspace", "WorkspaceError"]
