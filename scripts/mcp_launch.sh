#!/bin/sh
# Decipher MCP launcher. Dependency-free: starts the server when the
# environment is healthy, otherwise fails fast with a machine-readable
# bootstrap_required diagnostic on stderr (never a long build on stdio).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo '{"decipher_bootstrap_required": true, "reason": "missing_venv", "run": "sh scripts/bootstrap.sh"}' >&2
  exit 1
fi
if ! "$PY" -c "import mcp_server" >/dev/null 2>&1; then
  if ! PYTHONPATH="$ROOT/src" "$PY" -c "import mcp_server" >/dev/null 2>&1; then
    echo '{"decipher_bootstrap_required": true, "reason": "package_not_importable", "run": "sh scripts/bootstrap.sh"}' >&2
    exit 1
  fi
fi
cd "$ROOT"
exec env PYTHONPATH="$ROOT/src" "$PY" -m mcp_server "$@"
