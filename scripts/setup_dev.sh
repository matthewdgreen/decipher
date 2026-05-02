#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-"$ROOT/.venv/bin/python"}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python interpreter not found at $PYTHON" >&2
  echo "Create and activate .venv first, or set PYTHON=/path/to/python." >&2
  exit 2
fi

"$PYTHON" -m pip install -e "$ROOT"
"$ROOT/scripts/build_rust_fast.sh"

echo
echo "Decipher development setup complete."
echo "Check the install with:"
echo "  PYTHONPATH=src $ROOT/.venv/bin/decipher doctor"
