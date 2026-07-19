# Decipher MCP — Onboarding

The closed onboarding set for the Decipher MCP server. Read the section that
matches your role; after an investigation exists, `investigation_status` is the
authoritative live briefing (not this prose).

---

## 1. Operator quickstart

The three-line experience:

    git clone <this-repository>
    cd decipher
    codex          # or: claude
    > I would like to crack a cipher.

A checked-in `.mcp.json` (Claude Code) and `.codex/config.toml` (Codex, after
you trust the project) launch the server via `scripts/mcp_launch.sh`. The
launcher is dependency-free: when the environment is healthy it starts the
server; on a fresh clone it fails fast with a machine-readable
`bootstrap_required` diagnostic on stderr instead of running a long build over
stdio.

**Fresh clone:** run `sh scripts/bootstrap.sh` once. It creates `.venv`,
installs Decipher, builds the Rust kernels if `cargo` is present (degrades with
a note otherwise), and health-checks with `.venv/bin/decipher doctor --json`.
**After the first bootstrap, reconnect MCP or start a new session** so the
client re-launches the now-healthy server.

**Health check:** `.venv/bin/decipher doctor --json`.

**Where investigations live:** `~/.config/decipher/investigations/<id>/`
(override with `--registry-dir` or `$DECIPHER_MCP_REGISTRY`). Each directory is
self-contained and auditable; the server never garbage-collects.

**Picking up new code (sync & freshness, from inside a session):** when this
clone may be behind (e.g. a fix landed elsewhere), run — or ask the agent to
run — these two shell commands, then verify, then restart once if needed:

    git pull --ff-only
    sh scripts/bootstrap.sh    # fingerprint short-circuits when nothing changed

Verification: call `investigation_list` and compare its `server_code.git_head`
against `git rev-parse --short HEAD`. Match → the running server already has
the new code (it launched after the pull); carry on. Mismatch → the server
process predates the pull: restart the client session (the ONE step that
cannot be done from inside it) and re-check. This turns "did the changes
take?" into a one-call assertion instead of a guess.

**Recovery table:**

| Symptom | Fix |
|---|---|
| Launcher reports `missing_venv` / `bootstrap_required` | `sh scripts/bootstrap.sh`, then reconnect. |
| Server behavior predates a landed fix | Sync & freshness recipe above (`git pull` → bootstrap → `investigation_list` `server_code` check → restart session iff mismatch). |
| Project not trusted (Codex) | Trust the project, then reload so `.codex/config.toml` is read. |
| No API key | Everything works except independent verification; see §4 and the keyless walkthrough in the spec (Part 7.3). |
| Stale build (deps/kernels changed) | `sh scripts/bootstrap.sh` (the fingerprint short-circuits when nothing changed). |
| Interrupted investigation | It is on disk. `investigation_list` + `investigation_status` resume it from either client. |

---

## 2. Investigation methodology

You are the strategist of a cipher investigation; the Decipher MCP server is
your instrument bench and evidence store.

- **The brief is the source of truth.** `investigation_status` rebuilds the
  whole picture from server state every call — cipher, measured fingerprint,
  branch cards, hypothesis board, readings, experiments, evidence, and a
  rotating decode window. Never re-derive facts it already states; call it
  again whenever a result changes state (the `revision` you pass with mutating
  calls comes from it).
- **Work at the hypothesis level.** Form a hypothesis about the cipher family,
  create a branch for it (`hypothesis_branch_create`), test it, record evidence
  for and against, and reject it when settled. Keep several live candidates;
  `candidate_list` shows every branch with labeled signals — no single score
  defines the leader. Trust decoded text over any single number.
- **Measure before solving.** `observe_diagnosis` runs the LLM-free family
  diagnosis; an alphabet much larger than 26 means homophonic, and its
  constraint scores naturally sit below 1.0.
- **Long solver work is an experiment.** `experiment_submit` runs the no-LLM
  solver stack in the background; collect and, if good, install with
  `experiment_collect`. Never resubmit an identical config — duplicates are
  suppressed. Experiment types: `automated_solver` (family-routed general
  stack — substitution, homophonic, plain periodic, transform screens),
  `quagmire3_shotgun` (dedicated Rust keyword-alphabet search for
  Quagmire/keyed-tableau ciphers; use it when plain Vigenère-family search
  fails on a strongly periodic cipher — `automated_solver` cannot solve that
  family and will reject `cipher_system` hints naming it), and
  `composite_substitution_transposition` (peel-and-solve for a substitution
  THEN transposition — strong letter fit but no words → try the composite peel;
  `automated_solver` rejects `cipher_system` hints naming a substitution+
  transposition composite and redirects here). If `experiment_submit` does not
  advertise `quagmire3_shotgun` or `composite_substitution_transposition`, your
  server predates them — `git pull` this clone and restart the client session so
  a current server launches.
- **Read, then repair, then reverify.** When a candidate partly reads: record
  your reading (`reading_record`), compile concrete word hypotheses
  (`repair_hypotheses_test`), then ask the host to validate and install one
  winner (`repair_transaction`). The host rejects unsupported edits and any
  scoring regression; if you are confident the mechanical counter is wrong
  (correct words outside the common list), pass `verifier_arbitration=true` — an
  independent reader then arbitrates the repaired fork, and only a reading it
  judges strictly better installs; after two failed repair rounds on the same
  evidence it latches exhausted — broaden instead of polishing. Distributed
  damage that is a set of individually-simple key errors is still batch-repairable
  via `repair_hypotheses_test` → `repair_transaction`; do not treat `distributed`
  automatically as broaden-only (damage-scope routing is advisory — WF-4).
- **Verify before declaring.** `request_independent_verification` has a fresh
  reader judge your branch. Declaration (`meta_declare_solution`) is hard-gated
  on a positive fresh verification of the exact current content. Honest
  surrender (`meta_declare_unsolved`) is always available and never blocked.
- **Never stop without a verdict (WF-7).** Do not end a session holding an
  unverified candidate key. Before you stop — even stopping short of a solve —
  run `request_independent_verification` on your leading branch (the
  attestation is the record of how good it was, and your only independent
  signal for choosing between rival keys), then close with an explicit
  `meta_declare_solution` or `meta_declare_unsolved`. A session that ends with
  full keys, zero attestations, and no declaration leaves the investigation
  unmeasurable and unfinished.
- **Show the text.** When you report a verdict to the human — solved, partial,
  or unsolved — include the actual decode of your leading branch in the chat
  (paste `decode_show` output), together with its signals and the final
  attestation scalars. A verdict without the text forces the reader into the
  registry to see what you produced; the decode IS the deliverable, damaged
  or not.
- The host guidance block in the brief is advisory (policy ids shown); you may
  deviate with reason, except lines marked ENFORCED.

---

## 3. MCP capability reference (23 tools)

Kept in sync with `src/mcp_server/tools.py`; see the spec Part 5 for full
schemas.

**Create**
- `investigation_start` (create) — start a new investigation from inline ciphertext.

**Read**
- `investigation_list` (read) — list stored investigations, newest first.
- `investigation_status` (read) — the full self-briefing rebuilt from state.
- `observe_overview` (read) — compact measured facts for cheap re-orientation.
- `observe_diagnosis` (read) — ranked LLM-free cipher-family diagnosis.
- `decode_show` (read) — paired encoded/decoded rows for a branch.
- `hypothesis_next_steps` (read) — advisory next-step suggestions (WF-1).
- `candidate_list` (read) — the candidate portfolio with labeled signals (WF-6).
- `candidate_show` (read) — full detail for one candidate branch.
- `branch_adjudicate` (read) — read-only comparison table over 2–8 branches.

**Mutate**
- `hypothesis_branch_create` (mutate) — create a hypothesis branch.
- `hypothesis_branch_update` (mutate) — update a hypothesis branch's status/evidence.
- `hypothesis_branch_reject` (mutate) — mark a hypothesis rejected/superseded.
- `experiment_submit` (mutate) — queue a background automated-solver experiment.
- `experiment_collect` (mutate) — adjudicate/install a queued experiment.
- `reading_record` (mutate) — record your hash-bound reading of a branch.
- `comparison_record` (mutate) — record your ranking (best_partial vs accepts split).
- `repair_hypotheses_test` (mutate) — compile word hypotheses into scratch forks.
- `repair_transaction` (mutate) — host-validated install of one compiled winner (supports opt-in `verifier_arbitration`).
- `request_independent_verification` (mutate) — run a fresh independent reader.
- `act_set_model_variant` (mutate) — select the language-model variant.
- `meta_declare_solution` (mutate) — declare solved (hard-gated on verification).
- `meta_declare_unsolved` (mutate) — declare honestly unsolved (never gated).

---

## 4. Privacy and publication

Private ciphertext stays local: the registry lives under
`~/.config/decipher/`. Nothing is published or sent anywhere **except**
verify-episode candidate text: `request_independent_verification` sends the
candidate decode to the configured model provider so a fresh reader can judge
it. If that is unacceptable, run keyless — start the server with
`--verify-provider none` (or with no API key configured). Everything else works
keyless; only independent verification and the solved-declaration gate are
affected, and your strongest candidate stays available via `candidate_list`,
labeled "promising but not independently verified".

---

## 5. Contributing

- Preserve the investigation directory and the commit id when reporting.
- Redact ciphertext, keys, and candidate plaintext by default.
- Ask the user before opening any issue or PR.
- Use a separate branch/worktree for code changes; never develop on top of a
  live investigation.
