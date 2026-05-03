#!/usr/bin/env bash
# setup_dev.sh — one-shot development environment bootstrap.
#
# Creates .venv (if absent), installs the Python package in editable mode,
# and builds the required Rust extension (decipher_fast).
#
# Usage:
#   scripts/setup_dev.sh
#
# Override the Python interpreter (must be 3.11+):
#   PYTHON=/usr/local/bin/python3.11 scripts/setup_dev.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

die() {
  echo "" >&2
  echo "error: $*" >&2
  echo "" >&2
  exit 2
}

step() { echo ""; echo "==> $*"; }
info() { echo "    $*"; }

# ---------------------------------------------------------------------------
# Step 1 — Rust / cargo
# ---------------------------------------------------------------------------

step "Checking for Rust / cargo"
if ! command -v cargo >/dev/null 2>&1; then
  die "cargo not found.

  Install Rust (includes cargo) from https://rustup.rs/ :
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

  Then open a new shell (or run: source \"\$HOME/.cargo/env\") and rerun this script."
fi
info "cargo $(cargo --version 2>/dev/null | awk '{print $2}') found"

# ---------------------------------------------------------------------------
# Step 2 — find Python 3.11+
# ---------------------------------------------------------------------------

step "Locating Python 3.11+"

_py_ok() {
  # Returns 0 if $1 is an executable Python >= 3.11.
  [[ -n "$1" ]] && command -v "$1" >/dev/null 2>&1 \
    && "$1" -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null
}

if [[ -n "${PYTHON:-}" ]]; then
  _py_ok "$PYTHON" \
    || die "PYTHON=$PYTHON is not a Python 3.11+ executable."
  info "Using caller-supplied interpreter: $PYTHON"
else
  PYTHON=""
  for candidate in python3.11 python3.12 python3.13 python3.14 python3 python; do
    if _py_ok "$candidate"; then
      PYTHON="$candidate"
      break
    fi
  done

  if [[ -z "$PYTHON" ]]; then
    OS="$(uname -s)"
    case "$OS" in
      Darwin)
        INSTALL_HINT="    brew install python@3.11
    — or download an installer from https://www.python.org/downloads/"
        ;;
      Linux)
        if command -v apt-get >/dev/null 2>&1; then
          INSTALL_HINT="    sudo apt install python3.11 python3.11-venv python3.11-dev"
        elif command -v dnf >/dev/null 2>&1; then
          INSTALL_HINT="    sudo dnf install python3.11 python3.11-devel"
        else
          INSTALL_HINT="    Install python3.11 and python3.11-dev (or equivalent) for your distro."
        fi
        ;;
      *)
        INSTALL_HINT="    Download an installer from https://www.python.org/downloads/"
        ;;
    esac
    die "Python 3.11 or newer not found.

  Install it:
$INSTALL_HINT

  Then rerun this script (or set PYTHON=/path/to/python3.11)."
  fi

  info "Found $("$PYTHON" --version 2>&1)"
fi

# ---------------------------------------------------------------------------
# Step 3 — C build toolchain (needed by maturin / PyO3)
# ---------------------------------------------------------------------------

step "Checking for a C build toolchain"

if ! (command -v cc >/dev/null 2>&1 \
      || command -v gcc >/dev/null 2>&1 \
      || command -v clang >/dev/null 2>&1); then
  OS="$(uname -s)"
  case "$OS" in
    Darwin)
      TOOLCHAIN_HINT="    xcode-select --install"
      ;;
    Linux)
      PY_VER="$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
      if command -v apt-get >/dev/null 2>&1; then
        TOOLCHAIN_HINT="    sudo apt install build-essential python${PY_VER}-dev"
      elif command -v dnf >/dev/null 2>&1; then
        TOOLCHAIN_HINT="    sudo dnf install gcc gcc-c++ make python3-devel"
      else
        TOOLCHAIN_HINT="    Install gcc (or clang) and Python development headers for your distro."
      fi
      ;;
    *)
      TOOLCHAIN_HINT="    Install a C compiler and Python development headers for your platform."
      ;;
  esac
  die "No C compiler found (required by the Rust/PyO3 build).

  Install one:
$TOOLCHAIN_HINT

  Then rerun this script."
fi
info "C compiler found ($(cc --version 2>/dev/null | head -1 || echo 'ok'))"

# ---------------------------------------------------------------------------
# Step 4 — create .venv if absent
# ---------------------------------------------------------------------------

VENV="$ROOT/.venv"

step "Setting up virtual environment"
if [[ -x "$VENV/bin/python" ]]; then
  info ".venv already exists — skipping creation"
else
  info "Creating .venv with $("$PYTHON" --version 2>&1) ..."
  "$PYTHON" -m venv "$VENV"
  info ".venv created at $VENV"
fi

PYTHON="$VENV/bin/python"

# ---------------------------------------------------------------------------
# Step 5 — install Python package (editable)
# ---------------------------------------------------------------------------

step "Installing Python package (editable)"
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install -e "$ROOT"

# ---------------------------------------------------------------------------
# Step 6 — build Rust extension
# ---------------------------------------------------------------------------

step "Building Rust extension (decipher_fast)"
PYTHON="$PYTHON" "$ROOT/scripts/build_rust_fast.sh"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "=================================================="
echo " Decipher development setup complete."
echo "=================================================="
echo ""
echo " Quick check:   PYTHONPATH=src $VENV/bin/decipher doctor"
echo " Run tests:     PYTHONPATH=src $VENV/bin/python -m pytest tests/ -q"
echo ""
echo " To activate the venv in your current shell:"
echo "   source $VENV/bin/activate"
echo ""
