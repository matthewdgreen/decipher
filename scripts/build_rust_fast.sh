#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-"$ROOT/.venv/bin/python"}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python interpreter not found at $PYTHON" >&2
  echo "Create the virtualenv first, or set PYTHON=/path/to/python." >&2
  exit 2
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo was not found. Install Rust from https://rustup.rs/ and rerun." >&2
  exit 2
fi

"$PYTHON" -m pip install "maturin>=1.5,<2"
cd "$ROOT/rust/decipher_fast"
"$PYTHON" -m maturin develop --release

echo
echo "decipher_fast built successfully."
echo "Verify with:"
echo "  PYTHONPATH=src $ROOT/.venv/bin/decipher doctor"
