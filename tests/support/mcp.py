"""Shared fixtures for the MCP server tests (Part 11).

``make_server`` builds the server core directly (no subprocess) with a
synchronous experiment queue; ``call`` returns the parsed JSON body of one
tool call. ``english_basin`` white-boxes a damaged English substitution basin
onto an investigation's main branch so the repair tools have a changed fork to
work with (the 23-tool surface has no direct mapping tool — the same white-box
style the loop_v3 / hypothesis_actions tests use).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mcp_server.registry import InvestigationRegistry  # noqa: E402
from mcp_server.server import DecipherMCPServer  # noqa: E402

# A short English substitution: cipher symbol = lowercase of the plaintext
# letter. Word index 2 ("brown") is damaged (r -> X) so a "BROWN" hypothesis
# yields a changed fork.
BASIN_PLAINTEXT = "THE QUICK BROWN FOXES JUMPED"
BASIN_CIPHERTEXT = BASIN_PLAINTEXT.lower()
BASIN_DAMAGED_WORD_INDEX = 2
BASIN_CORRECT_WORD = "BROWN"


class DummyProvider:
    """A truthy stand-in that makes verification_available True; verify episodes
    are served by a registered fake session builder, never this object."""

    model = "dummy"
    provider_name = "dummy"


def make_server(tmp_path, *, verify="none", provider=None):
    registry = InvestigationRegistry(tmp_path / "registry")
    verify_provider = None if verify == "none" else (provider or DummyProvider())
    return DecipherMCPServer(
        registry=registry,
        verify_provider=verify_provider,
        synchronous_experiments=True,
    )


def call(server, name, **args):
    text, is_error = server.call_tool(name, args)
    body = json.loads(text)
    body["_is_error"] = is_error
    return body


def start(server, ciphertext=BASIN_CIPHERTEXT, **kw):
    return call(server, "investigation_start", ciphertext=ciphertext, **kw)


def apply_basin(runtime):
    """White-box: apply the correct key to main, damaging r -> X (word 2)."""
    ws = runtime.workspace
    alpha = ws.cipher_text.alphabet
    pt = ws.plaintext_alphabet
    for sym in alpha.symbols:
        upper = sym.upper()
        target = "X" if sym == "r" else upper
        ws.set_mapping("main", alpha.id_for(sym), pt.id_for(target))
