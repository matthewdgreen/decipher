"""Transport-neutral investigation service layer + operation manifest.

The service (:class:`~investigation_service.service.InvestigationService`) owns
the dispatch pipeline that both the MCP server and the structured investigation
CLI call into; the manifest (:mod:`investigation_service.manifest`) is the single
source of truth for the operation set both skins project from. See the interface
lockstep contract in ``docs/specs/investigation_cli_spec.md`` §6.
"""
from __future__ import annotations

from investigation_service.manifest import (
    OPERATIONS,
    ExternalEffect,
    OperationSpec,
    operation_for,
    render_tool_list,
    schema_for,
)
from investigation_service.service import (
    InvestigationService,
    LeasePolicy,
)

__all__ = [
    "OPERATIONS",
    "ExternalEffect",
    "OperationSpec",
    "operation_for",
    "render_tool_list",
    "schema_for",
    "InvestigationService",
    "LeasePolicy",
]
