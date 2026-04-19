"""Run-artifact schema for v2 agent runs.

A RunArtifact captures everything observable about a single agent run:
plan, branch tree, notebook, tool-call log, sub-agent transcripts, and
the final declare_solution payload. It is the research datum.
"""
from artifact.schema import (
    BranchSnapshot,
    NotebookEntry,
    RunArtifact,
    SolutionDeclaration,
    SubagentRun,
    ToolCall,
)

__all__ = [
    "BranchSnapshot",
    "NotebookEntry",
    "RunArtifact",
    "SolutionDeclaration",
    "SubagentRun",
    "ToolCall",
]
