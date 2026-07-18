# Session handoff — 2026-07-18

Self-contained context for starting a fresh session. Written at HEAD `ea78ddf`
(branch `main`, clean). Complements the persistent memory
(`~/.claude/.../memory/MEMORY.md` + `project_state.md`), the MCP proposal, the
provenance ledger, and the dogfood results log — this doc ties them together.

> **SAFETY-GATE NOTE (important):** examining decrypted **Borg / historical
> Latin manuscript plaintext** has repeatedly misfired an Anthropic safety gate
> and blocked the request. Do NOT read/print decoded plaintext, verifier
> `gloss`, or `anomalies` text for the historical pages. Scalar/enum metadata
> (char/word accuracy numbers, verifier confidence numbers, `damage_scope`,
> `reader_accepts_as_solution`, dict_rate/quad) is safe and sufficient for
> analysis. Synthetic-cipher plaintexts (original prose) are fine.

---

## 1. Where things stand

- **M5.3 (v3 control/repair reliability): COMPLETE.** All 6 slices landed +
  Fable-reviewed + Sequence A/B/C verified. See §3.
- **CLI-3 (human-first narrate display): LANDED** (`0ad06cb`, +`02d46c3`).
  Default output is a Claude-Code-style narrative; `-v` adds precise tool calls.
- **MCP dual-harness server: BUILT + LANDED end-to-end** (Phases 0→A→B→C).
  A `decipher mcp-serve` stdio server exposes 23 tools over the extracted
  `InvestigationHost`; onboarding files checked in; dogfooded live from Codex.
  See §4.
- **Verifier-arbitrated repair: LANDED** (`6330af6`) — closes the "reads
  correctly but can't persist the fix" wall found during dogfooding.
- **Full test suite: 1773 passed / 2 skipped** at HEAD. v3 loop unbroken
  throughout (every MCP step gated on an unchanged suite).
- **Open backlog: §6.** Headline items: composite substitution+transposition
  detection gap; verifier calibration on historical Latin; the Phase-D
  tooling-value experiment; M5.4 (repair reframe) still deferred.

## 2. This session's commit ledger (newest first)

```
ea78ddf docs: mark verifier-arbitrated repair fix landed
6330af6 Verifier-arbitrated repair acceptance (opt-in), from live MCP evidence
2112548 docs: round-4 live arms (MCP honest-fail + naive cDecryptor control)
1a5820b docs: round-4 self-test — composite sub+transposition gap
047b441 docs: MCP dogfood results log + repair-guard false-reject evidence
0dfef8a MCP server (Phases A-C): decipher mcp-serve + 23 tools + onboarding
02d46c3 CLI-3 follow-up: never truncate episode goals; raise digest caps
2a0af40 fix: normalize OpenRouter assistant history
5f269ac MCP: extract InvestigationHost from run_v3 (ZERO behavior change)
1171cb7 docs: specify zero-effort MCP onboarding UX
e6295c4 MCP Phase 0: policy-provenance ledger (66 controls)
533f60d MCP proposal: apply Fable review revisions + zero-effort onboarding
0ad06cb CLI-3: human-first narrate display; tool detail behind -v
9308a15 / 0092964  MCP proposal drafts
e725d27 Fix v3 null-mask candidate rendering
d7e1376 v3: bounded retry on transient 429 rate limits
10af31f v3 lead: bounded resilience to tool-less turns (K3 truncation)
93583d6 Self-heal a stale OpenRouter model cache in validate_model
616f297 Wire max_cost_usd through BenchmarkRunnerV2 to run_v3
2ca2c6d / a0ba63c / f371bba / 24541d2 / f0818cb / ce478db   M5.3 Slices 7/6/2+4/3/1/5
```
(`cc26506` M5.3a plan, `7715614`/`22b6656` harness hardening, `314b67d` plan
are also this arc.)

## 3. M5.3 recap (landed)

Six control/repair-reliability slices on the v3 agent loop:
- **Slice 1** (`f0818cb`): hard per-episode call budgets + per-run cost ceiling
  (`max_cost_usd`, wired to BenchmarkRunnerV2 in `616f297`).
- **Slice 3+B1** (`24541d2`): batched/cached `hypothesis_test_words`.
- **Slice 2+B2+4** (`f371bba`): durable repair-saturation state machine
  (process-vs-evidence failure taxonomy, `repair_exhausted` latch) +
  host-validated repair acceptance (8 ordered checks, default-deny).
- **Slice 6+B3** (`a0ba63c`): diplomatic verifier contract + **C6 reversal** —
  `meta_declare_solution` now requires a fresh POSITIVE attestation
  (`reader_accepts_as_solution`); weak/negative route repair/compare/broaden.
  One shared predicate `state.attestation_is_positive`.
- **Slice 7** (`2ca2c6d`): observability (4 branch roles), analyzer parity,
  trimmed regression fixture, and the Sequence-B replay test
  (`tests/test_v3_sequence_b.py`).
- Spec: `docs/specs/agent_v3_m5_3_control_reliability_spec.md` + per-slice impl
  specs. Sequence-C paid smoke passed (borg_0109v, gpt-5.5, 95.9%/82.1%, $1.29).

## 4. The MCP server (the session's centerpiece)

**What:** one `decipher-mcp` stdio JSON-RPC server (pure stdlib) exposing the v3
investigation surface as 23 tools, driven by Claude Code AND Codex on
subscription plans. The custom harness stays the firewalled benchmark
instrument. Rationale, decisions, and the falsifiable Phase-D experiment are in
`docs/mcp_dual_harness_proposal.md` (§1-§11).

**Architecture:**
- `src/investigation/host.py` — `InvestigationHost` (extracted verbatim from
  `run_v3`'s dispatch layer in `5f269ac`, byte-parity verified). Both `run_v3`
  and the MCP server drive the SAME host. `run_v3` (now 948 lines) is a thin
  turn-loop driver.
- `src/mcp_server/` (11 modules) — protocol, registry (revision-checked,
  flock writer-lease, state at `~/.config/decipher/investigations/<id>/`),
  the 23 tools, server-side verify, client-compiled repair.
- Gates enforced SERVER-SIDE per the provenance ledger: invariants +
  evidence-mechanisms hard; only 7 advisory softenings + 2 pre-registered
  divergences change form. Ledger: `docs/mcp_policy_provenance_ledger.md`
  (66 controls, updated for verifier-arbitration as REP-3/REP-4).

**Run it / onboard (checked in):** `.mcp.json` (Claude Code auto-discovers),
`.codex/config.toml` + `docs/prompts/decipher-investigate.md` (Codex),
`scripts/mcp_launch.sh` + `scripts/bootstrap.sh`, `docs/mcp_onboarding.md`,
`AGENTS.md` doctrine. Handshake test: pipe `initialize`+`tools/list` into
`.venv/bin/decipher mcp-serve --verify-provider none`. Fresh-clone simulation:
`tests/test_mcp_onboarding.py`.

**Keyless degradation:** without an API key the server works except
server-side verification; declaration stays gated (`no_verification_provider`).

## 5. Dogfood results (see `docs/evidence/mcp_dogfood_results.md` for detail)

Real Codex-over-MCP sessions cracking generated ciphers from a bare clone.
Original-prose synthetics (sealed answers in the session scratchpad; a new
session will need fresh ciphers/answers — the scratchpad is session-scoped).

- **Rounds 1-3 (mono / homophonic / no-boundary homophonic): 98-100% char.**
  Strong solves. The independent verifier repeatedly caught near-misses
  (BRESS→BRASS, homophonic residuals) and refused to over-declare.
- **TWO persistence failure modes found in the wild** ("reads correctly, can't
  persist the fix"): (A) repair guard false-rejected objectively-correct fixes
  on a collateral occurrence-count (TRAWLER not in the 5000-word list);
  (B) repair not attempted because a `distributed` verdict routed to broaden.
  → Motivated and FIXED by verifier-arbitrated repair (`6330af6`); mode B also
  got an AGENTS.md batch-repair doctrine line (WF-4 advisory).
- **Round 4 (substitution + columnar transposition): a real CAPABILITY GAP.**
  Isolated to detection/routing (de-transpose by hand → instant 100% solve).
  Diagnosis has no sub+transposition composite family and its
  `letters_substituted` atom SUPPRESSES transposition; the solver router's
  "no boundaries → homophonic" heuristic hijacks `transform_search` away from
  transpositions. Confirmed by both a $0 self-test AND a live MCP-Codex
  honest-fail. This is the INV "composite universally missed" finding at the
  tooling level.
- **Naive-Codex control (no MCP): cracked the round-3 homophonic 100% by
  cloning an external solver (cDecryptor).** Humbling for the raw-solver moat.
  Refined §6 read: the durable moat is EPISTEMIC DISCIPLINE (independent
  verifier + gate that convert "solver emitted X" into "reader accepts X"),
  effort/latency, and reproducible provenance — NOT raw cracking power.
- **Borg (real Latin manuscript) pages, benchmark-recorded accuracy:**
  - `borg_0109v`: consistently solved (median ~0.85 char, up to 0.968;
    64/129 recorded runs ≥0.85).
  - `borg_0171v`: hard-but-consistent (~0.91 automated anneal, 0.924 Sonnet;
    5/10 ≥0.85) — the recommended "hard Borg that solves."
  - `borg_0077v`: known-HARD (median 0.098; flagged "no automated solver yet
    acceptable"; one 0.841 Sonnet run). `borg_0045v`/`0140v`: inconsistent.
  - **NEW FINDING (verifier calibration on historical Latin), from
    investigation `c96916091a0f` (borg_0171v, Codex) — scalars only, no
    plaintext examined:** the user reports the RECONSTRUCTION was nearly
    perfect, yet the verify attestation was `reader_accepts_as_solution=False`,
    `target_language_confidence=0.82`, `semantic_recoverability=0.62`,
    `damage_scope=distributed`, `coherence=4`, and branch dict_rate/quad were
    None/low. → The diplomatic verifier AND the scalar quality metrics appear
    to UNDER-score correct-but-abbreviated 17th-c. manuscript Latin. Same class
    as 0077v. Likely causes: (a) benchmark char-scoring is against a bracketed
    editorial transcription (`d[u]cem`, `[drachmam s.]`) that a cipher solve
    can't reproduce; (b) the verifier is tuned toward fluent modern language and
    doesn't credit heavily-abbreviated historical Latin. Backlog item §6.

## 6. Open backlog (prioritized)

1. **Verifier + scalar-score calibration on historical (abbreviated) Latin.**
   Newest finding (`c96916091a0f`). The verifier rejects near-perfect Borg
   reconstructions; dict_rate/quad undervalue abbreviated Latin. Needs a
   Latin-aware readability signal and/or a de-bracketing scorer. High value —
   currently blocks closing real Borg pages through the verify→declare gate.
2. **Composite substitution+transposition detection/routing.** Add a composite
   family to the diagnosis, stop `letters_substituted` from suppressing
   transposition, and stop the "no boundaries → homophonic" router from
   hijacking `transform_search`. Capability exists once the transposition is
   peeled. (Round-4 gap.)
3. **Phase-D tooling-value experiment** (proposal §6): Claude Code vs Codex vs
   v3 on the agent-critical cases (0109v preflight-OFF, copiale_p017,
   Quagmire-3 synth), container-isolated. Decides how much custom loop to keep.
4. **M5.4 (repair reframe / interpretation packets):** still DEFERRED, gated on
   a $0 oracle reference-compiler experiment. The proposal recommends waiting
   for Phase-D before investing. Verifier-arbitrated repair already relieved the
   most acute persistence pain.
5. **M5.3a candidate-reliability follow-up:** planned in
   `docs/specs/agent_v3_m5_3a_candidate_reliability_spec.md` (`cc26506`).
6. Smaller: CLI-4 spinner (backlog in memory); the round-4 batch-repair for
   distributed-simple damage confirmation; Sequence D/E paid runs (user-gated).

## 7. How work gets done here (orchestration)

Per `CLAUDE.md`: **Fable (main loop) writes specs + oversees; Opus/Sonnet
sub-agents implement from written specs; Fable sub-agents review the diff;
commit once per reviewed slice.** Critical discipline: after every Fable
sub-agent, **verify it was served by `claude-fable-5`** (not gated to Opus) by
grepping the transcript JSONL:
```
grep -ho '"model": *"[^"]*"' <subagents/agent-<id>.jsonl> | sort | uniq -c
```
Every review this session was verified fully `claude-fable-5`. Landing pattern:
coder works in a **manually-created worktree at the right base** (the Agent
`isolation:worktree` flag misfired once to a stale base — prefer manual
`git worktree add`), then `git diff <base> --binary | git apply` onto main,
full suite, Fable review, commit. Several transient API stalls occurred; resume
sub-agents via SendMessage or re-dispatch fresh.

## 8. Models / keys / config

- **Confirmed agent model: `gpt-5.5` (OpenAI)** for decipher runs; billed to the
  OpenAI account (`.decipher_keys/openai_api_key`). `--model gpt-5.5`
  auto-routes. gpt-5.6 tiers usable (luna = cheap episode tier).
- OpenRouter: any `/`-model auto-infers OpenRouter; key at
  `.decipher_keys/openrouter_api_key`. `validate_model` now self-heals a stale
  pricing cache (`93583d6`). **Kimi K3 (`moonshotai/kimi-k3`) tool-calls fine**
  — its earlier failures were output-token truncation (fixed `10af31f`) and a
  stale cache, NOT format problems.
- Anthropic: keychain `service=decipher`; modest credits. Sonnet 4.6 is the
  strong historical-manuscript model in recorded results.
- Full suite: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` (~4 min,
  1773/2). Venv Python 3.11 at `.venv/`.

## 9. Housekeeping notes

- **Stale worktrees** to consider removing (not from active work):
  `.claude/worktrees/cli3_manual` (CLI-3 already landed),
  `borg-word-boundary-rescore`, `heuristic-neumann-e6cda2` (predate this
  program). `git worktree remove --force <path>` when confirmed unwanted.
- Reference docs map: proposal `docs/mcp_dual_harness_proposal.md`; ledger
  `docs/mcp_policy_provenance_ledger.md`; onboarding `docs/mcp_onboarding.md`;
  specs under `docs/specs/` (m5_3*, mcp_host_extraction, mcp_server,
  verifier_arbitrated_repair, m5_3a); evidence + running results under
  `docs/evidence/`.
- The MCP investigation registry lives OUTSIDE the repo at
  `~/.config/decipher/investigations/<id>/` (investigation.json + meta.json +
  events.jsonl + lease.lock). Not versioned.
