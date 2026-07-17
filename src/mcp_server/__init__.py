"""Decipher MCP stdio server (Phases A-C).

Exposes the v3 investigation surface (``InvestigationHost`` + the composite
dispatcher + the verify dispatcher) as a tools-only MCP server over
newline-delimited JSON-RPC. Pure stdlib protocol layer; the domain logic is
imported from ``investigation`` (never duplicated). See
``docs/specs/mcp_server_spec.md``.
"""
