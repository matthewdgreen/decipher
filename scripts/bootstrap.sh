#!/bin/sh
# Decipher clean-clone bootstrap (idempotent, approval-first). Prepares a fresh
# checkout so the MCP launcher can start the server: creates .venv, installs
# Decipher, builds the Rust kernels when cargo is present (degrades otherwise),
# and health-checks with `decipher doctor --json`. Emits machine-readable JSON.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1. Serialize concurrent runs with an atomic mkdir lock (stale after 30 min).
LOCK="$ROOT/.bootstrap.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +30 2>/dev/null)" ]; then
    rmdir "$LOCK" 2>/dev/null || true
    if ! mkdir "$LOCK" 2>/dev/null; then
      echo '{"bootstrap": "locked"}' >&2
      exit 1
    fi
  else
    echo '{"bootstrap": "locked"}' >&2
    exit 1
  fi
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

# 2. Find Python >= 3.11 (never invoke sudo or a package manager).
PYBIN=""
for cand in python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      PYBIN="$cand"
      break
    fi
  fi
done
if [ -z "$PYBIN" ]; then
  echo '{"bootstrap": "failed", "layer": "prerequisite", "missing": "python>=3.11", "install": "https://www.python.org/downloads/ or brew install python@3.11"}' >&2
  exit 1
fi

# 3. Build fingerprint over pyproject.toml + Cargo.lock; short-circuit if cached.
if command -v shasum >/dev/null 2>&1; then
  HASH="shasum -a 256"
else
  HASH="sha256sum"
fi
FP="$(cat "$ROOT/pyproject.toml" "$ROOT/rust/decipher_fast/Cargo.lock" 2>/dev/null | $HASH | awk '{print $1}')"
FPFILE="$ROOT/.venv/.decipher_build_fingerprint"
if [ -f "$FPFILE" ] && [ "$(cat "$FPFILE" 2>/dev/null)" = "$FP" ]; then
  if "$ROOT/.venv/bin/decipher" --help >/dev/null 2>&1; then
    echo '{"bootstrap": "ok", "cached": true}'
    exit 0
  fi
fi

# 4. Create the venv if needed.
[ -d "$ROOT/.venv" ] || "$PYBIN" -m venv "$ROOT/.venv"

# 5. Install Decipher + dependencies.
if ! "$ROOT/.venv/bin/python" -m pip install -q -e ".[providers,dev]" >"$ROOT/.bootstrap.piplog" 2>&1; then
  echo '{"bootstrap": "failed", "layer": "python_deps"}' >&2
  tail -n 15 "$ROOT/.bootstrap.piplog" >&2
  exit 1
fi

# 6. Rust kernels (required accelerator) — build when cargo is present, else degrade.
if command -v cargo >/dev/null 2>&1; then
  "$ROOT/.venv/bin/python" -m pip install -q maturin >/dev/null 2>&1 || true
  if ! ( cd "$ROOT/rust/decipher_fast" && "$ROOT/.venv/bin/python" -m maturin develop --release ) >"$ROOT/.bootstrap.rustlog" 2>&1; then
    echo '{"bootstrap": "failed", "layer": "rust_build"}' >&2
    tail -n 15 "$ROOT/.bootstrap.rustlog" >&2
    exit 1
  fi
else
  echo '{"bootstrap": "degraded", "layer": "prerequisite", "missing": "cargo", "install": "https://rustup.rs (curl https://sh.rustup.rs -sSf | sh)", "note": "continuing without Rust kernels; some solvers are slow/unavailable"}' >&2
fi

# 7. Record the fingerprint and health-check.
printf '%s' "$FP" > "$FPFILE"
if ! "$ROOT/.venv/bin/decipher" doctor --json >/dev/null 2>&1; then
  echo '{"bootstrap": "failed", "layer": "health_check"}' >&2
  exit 1
fi

# 8. Done.
echo '{"bootstrap": "ok"}'
exit 0
