"""`python -m mcp_server` entry point (Part 1.2).

Parses the same four flags as `decipher mcp-serve` and calls the same
``serve_stdio`` — a stable entry that does not require the console script, so a
bare source checkout (or the fresh-clone launcher) can start the server.
"""
from __future__ import annotations

import argparse

from mcp_server.server import serve_stdio


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mcp_server")
    parser.add_argument("--registry-dir", default=None)
    parser.add_argument(
        "--verify-provider", default="auto",
        choices=["auto", "anthropic", "openai", "gemini", "openrouter", "ollama", "none"],
    )
    parser.add_argument("--verify-model", default=None)
    parser.add_argument("--max-cost-usd", type=float, default=5.0)
    args = parser.parse_args()
    serve_stdio(
        registry_dir=args.registry_dir,
        verify_provider=args.verify_provider,
        verify_model=args.verify_model,
        max_cost_usd=args.max_cost_usd,
    )


if __name__ == "__main__":
    main()
