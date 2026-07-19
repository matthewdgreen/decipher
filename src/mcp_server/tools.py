"""MCP tool surface — a thin projection of the shared operation manifest.

The tool definitions and the parallel ``TOOL_CLASSES`` map that used to live
here are folded into the single operation manifest
(:mod:`investigation_service.manifest`), consumed by BOTH the MCP server and the
structured investigation CLI (interface lockstep contract, spec §6). This module
re-exports the MCP-facing projections so existing ``mcp_server.tools`` importers
keep working byte-for-byte.
"""
from __future__ import annotations

from investigation_service.manifest import (  # noqa: F401
    MCP_TOOL_DEFINITIONS,
    OPERATIONS,
    TOOL_CLASSES,
    TOOL_NAMES,
    ExternalEffect,
    OperationSpec,
    render_tool_list,
    schema_for,
)

__all__ = [
    "MCP_TOOL_DEFINITIONS",
    "OPERATIONS",
    "TOOL_CLASSES",
    "TOOL_NAMES",
    "ExternalEffect",
    "OperationSpec",
    "render_tool_list",
    "schema_for",
]
