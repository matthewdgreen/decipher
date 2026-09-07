# Packaging and distribution plan — PyPI wheel, security hygiene, README rewrite

**Status: PARKED (written 2026-09-07).** This plan is not on the current
development queue. `docs/improvement_program_plan.md` ("Current plan —
2026-09-06", R0→R4) and `TODO.md` remain the schedule. When this program is
scheduled, add a pointer in `TODO.md` and treat this file as the governing
order. Nothing here changes investigation doctrine, runtime gates, or
authorizes paid runs.

## Why

Today the only install path is clone → `sh scripts/bootstrap.sh` → venv →
Rust build. For an AI coding agent on an arbitrary machine that is a lot of
surface to get wrong, and a cautious agent audits it before running it (the
2026-07 external evaluation balked at exactly this). A prebuilt wheel with the
Rust kernel bundled, published with verified provenance, gets most of what a
native binary would without conceding the Python design. It is also the
prerequisite for an MCP registry entry, which is the discovery path agents
actually use.

Three things have to land together for that wheel to be trusted rather than
flagged: the package must install cleanly (namespace refactor), it must not
look like credential harvesting or an opaque bulk download (security hygiene),
and the README must read as documentation rather than as instructions aimed
at agents (README rewrite).

## Facts that shape the plan (verified 2026-09-07)

- **The PyPI name `decipher` is taken** (Forsta survey REST client, v29.2.1,
  released 2026-01, actively maintained). A new distribution name is needed.
- **`src/` is a flat namespace.** Setuptools would install `agent`, `analysis`,
  `artifact`, `automated`, `benchmark`, `ciphers`, `experiments`, `external`,
  `frontier`, `investigation`, `investigation_service`, `mcp_server`,
  `models`, `ocr`, `preprocessing`, `services`, `testgen`, `workspace`, `cli`,
  and `investigation_cli` as **top-level** modules. Several of those names will
  collide on real machines. ~1,700 import sites across `src/`, `tests/`,
  `scripts/`; 45 docs/scripts reference the `PYTHONPATH=src` convention.
- **Models are 589 MB** (13 × 47.5 MB `models/ngram5_*.bin`), tracked in git.
  PyPI's default per-file limit is 100 MB. Every model already has a sidecar
  `*.bin.metadata.json` with a `sha256` field and a `redistribution_status`.
- **Runtime data is only `models/` and `resources/`** (2 MB). `corpora/` and
  `corpus_data/` (9 GB) are referenced only by `src/testgen/corpus_library.py`.
- **The Rust kernel** is a single pyo3 0.21 `cdylib` (`rust/decipher_fast`,
  ~4.3k lines) with pure-Rust deps (rand, rayon, sha1). It does not use
  `abi3` today, so a wheel would be per-Python-version. There is no
  `.github/workflows/` directory.
- **Data paths are anchored on `Path(__file__).resolve().parents[2]`**
  (`src/analysis/model_registry.py:84`, `src/analysis/dictionary.py:31`),
  i.e. the repo root. Wrong inside an installed wheel.
- **Key lookup scans broadly** (`src/cli.py:85-125`): env var, then
  `.decipher_keys/<provider>_api_key` in the repo root **and in the current
  working directory**, then the OS keychain, for four providers.
- **POSIX-only code:** `fcntl.flock` lease lock (`src/mcp_server/registry.py:212`,
  guarded import, no lock on the non-POSIX branch); `select.select` on a pipe
  fd plus `pass_fds` / `start_new_session` in `--detach`
  (`src/investigation_cli.py:658,697`); `sys.executable -m cli` re-exec
  (`src/investigation_cli.py:697`); `SIGTERM` handler
  (`src/investigation_service/service.py:599`).
- **`run_python`** exists in the v2 tool list (`src/agent/tools_v2.py`) but is
  **not** exposed through the MCP server. Worth stating in docs.
- The MCP server is hand-rolled JSON-RPC over stdio (no `mcp` dependency);
  `decipher mcp-serve` already exists, so `scripts/mcp_launch.sh` is not
  load-bearing for a packaged install.

## Goals

1. `pip install <name>` / `uvx <name> mcp-serve` works on every likely
   platform with the Rust kernel included, no compiler, no shell scripts.
2. A security-minded reviewer (human or agent) finds nothing surprising:
   verified provenance, minimal core deps, explicit key handling, hashed model
   downloads, a written statement of what the package touches.
3. The README is documentation for humans that agents can also read, and
   the PyPI page (which is the README) carries enough prose that a search for
   the tool's problem space finds it.

## Non-goals

- A native GUI or a single-file `.exe`. The MCP client is the front end.
- Docker as an install path. Revisit only if the wheel matrix leaves gaps.
- Changing solver behavior, agent doctrine, or the operation manifest.

## Target platforms

One wheel per OS × architecture, tagged `cp311-abi3` (covers Python 3.11+),
plus an sdist. Rust deps are pure Rust, so cross-compilation is uneventful.

| Tag | Runner | Notes |
|---|---|---|
| `manylinux_2_28_x86_64` | ubuntu-latest | the common agent sandbox |
| `manylinux_2_28_aarch64` | ubuntu-24.04-arm (or QEMU via maturin-action) | Graviton / ARM containers |
| `musllinux_1_2_x86_64` | ubuntu-latest | Alpine-based containers |
| `musllinux_1_2_aarch64` | ubuntu-24.04-arm | Alpine on ARM |
| `macosx_11_0_arm64` | macos-latest | Apple silicon |
| `macosx_10_12_x86_64` | macos-13 | Intel Macs |
| `win_amd64` | windows-latest | Claude Desktop on Windows launches MCP natively |
| `win_arm64` | windows-11-arm | Copilot+ / Windows-on-ARM; cheap to add |

Python floor stays 3.11 (`requires-python = ">=3.11"`). `abi3-py311` means a
3.14 user gets the same wheel; no rebuild per Python release.

## Phases

Each phase is one spec under `docs/specs/`, implemented by a coding
sub-agent, reviewed by a Fable sub-agent (with the model-served check from
`CLAUDE.md`), and committed as one commit. Phases P1–P3 are independent of
each other and can run in parallel after P0.

### P0 — Distribution name and namespace refactor

**Decision needed from the owner:** the PyPI distribution name. Avoid anything
adjacent to the existing `decipher` package (a `decipher-cipher` next to
`decipher` trips typosquat heuristics). Candidates: `decipher-cryptanalysis`,
or a fresh name. The **import package** becomes `decipher` regardless (the
import name does not have to match the distribution name, and `decipher`
is not taken as an import name by the survey client — verify this on PyPI
before finalizing, and choose a different import name if it is).

Work:
- Move `src/<pkg>/` → `src/decipher/<pkg>/` for all 18 packages; `src/cli.py`
  → `src/decipher/cli.py`; `src/investigation_cli.py` →
  `src/decipher/investigation_cli.py`.
- Rewrite imports mechanically (`from analysis.x` → `from decipher.analysis.x`,
  etc.) across `src/`, `tests/`, `scripts/`. Use a script (e.g. rope, or a
  `sed` over the 20 known prefixes) and then run the full suite.
- Entry point becomes `decipher = "decipher.cli:main"`; `python -m
  decipher.mcp_server` replaces `python -m mcp_server`.
- Delete every `PYTHONPATH=src` reference in `scripts/`, `docs/`, `README.md`,
  `AGENTS.md`, `CLAUDE.md`, `.codex/config.toml`, `.mcp.json`, `pyproject.toml`
  pytest config. An editable install (`pip install -e .`) makes the package
  importable without it.
- `src/decipher/__init__.py` exposes `__version__` from package metadata.
- Update `CLAUDE.md` "Key Files" tree and `docs/test_inventory.md` paths.

Required tests: full suite green under `pip install -e .` with no
`PYTHONPATH`; `tests/test_interface_parity.py` unchanged in behavior; a new
`tests/test_package_layout.py` asserting no top-level module other than
`decipher` is importable from the installed package (walk `site-packages`
for the dist's `RECORD`).

### P1 — Security hygiene

Absorbs the parked "supply-chain hygiene" item from 2026-07-19.

- **Dependency extras.** Core = `numpy`, `rich` (and `Pillow` only if OCR is
  core; otherwise move it to `[ocr]`). `[agentic]` = `anthropic`, `openai`,
  `google-genai`, `keyring`. `[dev]` = `pytest`, `maturin`. Every local /
  keyless path (`diagnose`, `crack` without `--agentic`, `mcp-serve`,
  `investigation`) must import cleanly on the core install; `verify` degrades
  to the existing structured `no_verification_provider` refusal.
- **Explicit key handling.** Replace the scan in `src/cli.py` with: (1) the
  provider-specific env var; (2) `DECIPHER_KEY_FILE` or a file under the
  data dir (P2) named for the provider; (3) OS keychain **only** when the
  user opted in (`--keyring` flag or a config setting), and only for the
  provider actually selected. Remove the current-working-directory scan.
  Log which source supplied a key (never the key).
- **Lock file.** Commit a `uv.lock` (or `requirements-lock.txt` from
  pip-compile) used by CI and by `scripts/bootstrap.sh`; keep the
  fingerprint short-circuit working.
- **`SECURITY.md` / "What this package touches".** Short, factual: no
  telemetry; network egress only to the configured LLM provider, the
  OpenRouter pricing endpoint (when that provider is used), and the model
  download host; files written only under the data dir and paths the user
  names; `run_python` is a v2 CLI-agent tool, not exposed via MCP; how to
  report a vulnerability. Link it from the README.
- **Subprocess statement.** Document that `investigation --detach` spawns a
  worker of the same package, and that the v2 `run_python` tool runs the
  interpreter that is running Decipher.

Required tests: import-time test that core modules do not import `anthropic`,
`openai`, `google`, or `keyring`; key-resolution unit tests covering each
source and asserting no cwd lookup; a test that the keychain is not queried
without opt-in (monkeypatch `keyring.get_password` to raise).

### P2 — Data directory and model fetch

- Introduce one resolver, `decipher.paths`, with: `DECIPHER_DATA_DIR` env →
  platform default (`platformdirs` user data dir, or a hand-rolled
  equivalent: `~/.local/share/decipher`, `~/Library/Application
  Support/decipher`, `%LOCALAPPDATA%\decipher`) → repo-root `models/` when
  running from a checkout. `model_registry.py` and `dictionary.py` use it.
  `resources/` (2 MB) ships inside the wheel as package data.
- `decipher models list|fetch|verify`. A checked-in manifest
  (`src/decipher/models_manifest.json`) lists every redistributable model with
  URL, size, and the `sha256` already present in each sidecar. `fetch`
  downloads to the data dir, verifies the hash before the file is renamed
  into place, and prints source and destination. **No implicit downloads.**
  When a model is missing at runtime, fail with the exact `fetch` command.
- Hosting: GitHub release assets on the tagged release (simplest, no new
  account) or a Hugging Face dataset repo. Decision for the owner; either
  works with the manifest. Only models whose sidecar says
  `redistribution_status: redistributable` go in the manifest.
- `decipher doctor` reports the data dir, which models are present and
  verified, and the kernel status.

Required tests: resolver precedence; fetch verifies hash and refuses a
mismatch (serve a bad file from a local HTTP server in the test); the MCP
intake and `diagnose` work with zero models present.

### P3 — Windows portability

- `src/mcp_server/registry.py`: on non-POSIX use `msvcrt.locking`, or replace
  the lease lock with an atomic-`mkdir` lock (the pattern `bootstrap.sh`
  already uses) on all platforms.
- `src/investigation_cli.py` `--detach`: replace `select` on a pipe fd with a
  reader thread over `proc.stdout`; drop `pass_fds` in favor of passing the
  handshake through the child's stdout; use `start_new_session` on POSIX and
  `creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS` on Windows.
- Worker re-exec: build argv as `[sys.executable, "-m", "decipher.cli", ...]`
  (P0 makes this valid on an installed package).
- `service.py`: register `SIGTERM` only when `hasattr(signal, "SIGTERM")` and
  the platform delivers it; on Windows handle `SIGBREAK`.
- Replace every user-facing message that names a `.sh` script with the
  packaged equivalent (`decipher doctor`, `pip install <name>`).

Required tests: the existing suite green on the `windows-latest` CI job (P4
provides it); a `--detach` round-trip test that runs on all three OSes.

### P4 — abi3 wheel and CI matrix

- `rust/decipher_fast/Cargo.toml`: `pyo3 = { version = "0.21", features =
  ["extension-module", "abi3-py311"] }`.
- Single mixed wheel via maturin: in the root `pyproject.toml` set
  `build-backend = "maturin"`, `[tool.maturin] python-source = "src"`,
  `manifest-path = "rust/decipher_fast/Cargo.toml"`, `module-name =
  "decipher._fast"`. `polyalphabetic_fast.py` imports `decipher._fast` and
  keeps a `decipher_fast` alias for one release for anyone with the old
  venv layout. Drop `rust/decipher_fast/pyproject.toml`.
- `.github/workflows/wheels.yml` using `PyO3/maturin-action` over the table
  above; `sdist` job; a `test` job that installs each built wheel into a
  clean venv and runs `pytest` (core install first, then `[agentic]`).
- `.github/workflows/publish.yml`: on a `v*` tag, download artifacts and
  publish with **PyPI Trusted Publishing** (OIDC, no stored token) and
  `attestations: true` (PEP 740). Publish to TestPyPI first from a
  `rc` tag.
- `scripts/bootstrap.sh` becomes the *development* bootstrap only; the README
  no longer shows it as the install path.

Required tests: the CI matrix itself; a smoke test in each wheel job that
imports `decipher._fast` and runs `decipher doctor --json`.

### P5 — MCP launch path and registry

- `.mcp.json` and `.codex/config.toml` (developer checkouts) keep the
  checkout launcher. Documented end-user config becomes:
  `{"command": "uvx", "args": ["<name>", "mcp-serve"]}` and the `pipx`
  equivalent.
- Add `server.json` for the MCP registry (registry.modelcontextprotocol.io)
  referencing the PyPI package; publish after the first stable release.
- `docs/mcp_onboarding.md`: replace bootstrap instructions with the packaged
  path; keep the checkout path in a "developing Decipher" section.

### P6 — README rewrite

This is the I-7 capstone rewrite already required by the investigation-CLI
program, scoped here for packaging. Current README is 943 lines with ~20
sections mixing user, agent, benchmark, and developer material.

Target structure (aim for ≤ 350 lines at top level, with everything else
linked into `docs/`):

1. **One paragraph** saying what it is, in the words people search for:
   classical cipher cryptanalysis, monoalphabetic and homophonic substitution,
   Vigenère/Quagmire, transposition, book ciphers; historical manuscripts
   (Borg, Copiale), Zodiac 408, Kryptos-style problems; statistical
   diagnosis and an MCP server for AI agents.
2. **Install** — `pip install <name>` / `uvx`, supported platforms table,
   `decipher models fetch`, `decipher doctor`.
3. **Use it from an AI agent (MCP)** — the config snippet, a two-line
   example session, link to `docs/mcp_onboarding.md`.
4. **Use it from the command line** — `diagnose`, `crack`, `investigation`.
5. **What it touches** — three sentences plus a link to `SECURITY.md`.
6. **Agentic solving (optional, needs an API key)** — extras, providers,
   cost note, link to detailed doc.
7. **Benchmarks and results** — headline numbers with links to the reports.
8. **Developing Decipher** — clone, `scripts/bootstrap.sh`, Rust build,
   tests; link to `CLAUDE.md`/`AGENTS.md`.
9. License, attribution.

**Rewrite the "Note for AI coding agents" block.** Text addressed to agents in
imperative voice has the shape of prompt injection, and well-behaved agents
are told to treat instructions inside content as data. Replace it with a
plain "Recommended workflow" paragraph written for humans: the built-in MCP
server and CLI cover diagnosis, solving, candidate persistence, and
verification; external solver frameworks are not needed for those steps. Same
information, no directive.

Move out of the README: testgen, frontier/parity runs, errata management,
language-model building, parallelism details, the long agentic flag reference.
Each gets its own `docs/` page (several already exist).

Required review: a Fable review that reads the README cold, as a first-time
user and as an agent deciding whether to install, and reports anything it
would flag.

### P7 — Release checklist

1. `rc` tag → TestPyPI → install from TestPyPI on macOS, Linux, Windows →
   `decipher doctor`, `decipher models fetch en`, `diagnose` on a fixture,
   `mcp-serve` handshake from Claude Desktop config.
2. `v0.2.0` tag → PyPI with attestations → confirm the "verified" provenance
   badge on the PyPI page.
3. Submit `server.json` to the MCP registry.
4. Update `CLAUDE.md` (install path, key files tree), `TODO.md`, and close
   this plan.

## Division of labor

Per `CLAUDE.md`: specs by Fable in the main session; P0 and P4 to Opus
sub-agents (multi-file, behavioral); P1–P3 to Opus; P5 to Sonnet; P6 drafted
by Fable, reviewed cold by a second Fable sub-agent; every review runs the
model-served check. P0 must land and be committed before anything else
starts, because every other phase edits paths that P0 moves.

## Open decisions for the owner

1. Distribution name (and import name if `decipher` turns out to be taken as
   an import name).
2. Model hosting: GitHub release assets vs. Hugging Face.
3. Whether `Pillow`/OCR stays in the core install.
4. Whether to keep `musllinux` and `win_arm64` in the first release or add
   them in a follow-up (cost is one matrix row each; recommendation: keep).

## Effort

Rough, sequential: P0 two to three days including the suite; P1–P3 one to
two days each; P4 two days including the first green matrix; P5 half a day;
P6 two days including the cold review; P7 one day. Most risk is in P0 (scale)
and P4 (first-time CI on three OSes).
