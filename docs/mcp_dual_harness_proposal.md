# Proposal: Decipher as an MCP Tool Server for Claude Code and Codex

Status: PROPOSAL — no implementation. Written 2026-07-17 from the harness
retrospective discussion (M5.3 completion week). Revised after Codex review to
add policy provenance, a technical firewall, concurrency rules, and a staged
decision experiment. Decision owner: Matthew.

---

## 1. Summary

Expose decipher's investigation surface as a **single MCP server** consumable
by both **Claude Code** (Claude Max subscription) and **Codex CLI** (ChatGPT
Pro subscription), while keeping the existing custom harness as the
**benchmark/evaluation instrument**. Frontier-harness sessions become the
vehicle for exploratory and real-unknown decipherment through prepaid
subscription capacity; the custom v2/v3 harness remains the controlled,
ground-truth-firewalled lab bench for scored model science.

MCP is the capability, state, and security boundary. Client-specific skills
or standing instructions supply the investigative methodology. They are
complementary, not competing alternatives.

The proposal includes a cheap, falsifiable experiment that decides how much
custom loop we should keep maintaining. **No existing v3 nudge, gate, or
barrier is removed in MCP v1 merely because it looks inelegant.** Phase 0 first
records why each control exists, the artifact or test that motivated it, and
the evidence required to soften or retire it.

## 2. Motivation

### 2.1 The honest harness assessment

The week of M5.3 produced clear evidence of **convergent evolution**: much of
our recent engineering re-implements what mature agent harnesses already
provide.

Built or fixed by hand, recently, in our loop:

- per-run cost ceilings and per-episode call budgets (M5.3 Slices 1);
- 429 rate-limit retry with Retry-After handling (K3 incident);
- output-truncation resilience for reasoning models (K3 incident);
- tool-less-turn nudges;
- a background compute queue (M4 experiments) = "background tasks";
- fresh-context worker episodes (M2) = "subagents";
- rebuilt-context state management (M1) = "context management";
- a record/replay harness (Slice 7);
- a human-first CLI display whose spec literally says "model this after
  Claude Code" (CLI-3, in flight).

Each carries a recurring maintenance tax. Some are generic harness features;
others encode hard-won responses to observed model failures and cannot safely
be discarded without an ablation. The durable IP is the ~95 domain tools
(solvers, scorers, segmentation, zenith SA, diagnosis panels, dictionaries,
repair menus), the benchmark + scorer + firewall, the serializable evidence
state, and the M5/M5.3 epistemic gate machinery.

The MCP hypothesis is therefore not "a mature harness will naturally behave
well." It is narrower: a mature harness may plan and recover better when given
Decipher's improved high-level tools and evidence state, while the server
continues enforcing the invariants that experiments proved necessary.

### 2.2 What the custom harness genuinely buys

1. **Ground-truth firewall.** The host constructs every model-visible byte;
   `tests/test_ground_truth_firewall.py` proves the agent cannot see
   plaintext. Indispensable for *scored benchmark* runs; near-impossible to
   certify in a general harness with filesystem access.
2. **Controlled comparison.** Fixed substrate (context assembly, tools,
   replayable transcripts, diffable artifacts) is what makes bake-offs valid.
   General harnesses update weekly and are tuned for their own vendor's
   models — using one as the bench confounds model with harness.
3. **Adversarial epistemic gates as host code.** Verification-gated
   declaration (hash-bound positive attestation), host-validated repair
   acceptance (reject-before-install, default-deny), saturation latches. In a
   model-driven loop these degrade into suggestions unless enforced
   server-side.
4. **Cost economics** (weakest pillar): v3's rebuilt-context runs are ~60%
   cheaper per run than transcript accumulation — but see §3: subscriptions
   mostly delete this argument for exploratory work.

### 2.3 The economic driver

Metered API spend to date is concentrated in *runtime*, not orchestration:
M6 bake-off ~$75, Stage-1 packets ~$10–12, individual agentic runs $1.3–4.6.
Meanwhile the orchestration work (spec/review/coding agents in Claude Code)
already rides a flat-rate subscription.

- **Claude Max** covers Claude Code usage → Claude-led investigation becomes
  flat-rate.
- **ChatGPT Pro** covers Codex CLI (signed in with the ChatGPT account) →
  GPT-led investigation becomes flat-rate. This is the *stronger* half:
  **gpt-5.5 is the project's confirmed best agent model**, so the best-known
  configuration moves to prepaid capacity.
- Both subscriptions meter capacity in rate windows (5-hour/weekly caps):
  exploration is effectively flat-rate; **batch sweeps are not** and stay on
  API billing.

These are July 2026 operating assumptions, not architectural guarantees.
Provider entitlements, model availability, and capacity limits must be checked
again before Phase D; the design should still make sense if either subscription
becomes less favorable.

### 2.4 The Borg artifact is evidence for both sides

The local M5.3 artifact investigation should prevent an overly simple reading
of this proposal:

- A null-mask experiment generated a 96.5% character / 82.1% word candidate.
- The host renderer erased canonical word boundaries, causing the verifier to
  judge a fused string rather than the actual candidate. That was a host bug.
- Even on the fused text, the compare worker ranked the null-mask branch first,
  but returned `winner=null` because "best partial" and "solved winner" were
  conflated. The host then discarded the ranking. That was a contract bug.
- The fixed rerun never attempted the successful sparse-null configuration and
  spent its budget on repeated simple-substitution/transform variants. That was
  a genuine agent-policy failure; existing coverage nudges arose for exactly
  this class of behavior.

The lesson is not "remove host policy." It is to separate host-owned facts and
invariants from model-owned investigative judgment, preserve the provenance of
every policy, and test changes one by one.

## 3. Proposed architecture

### 3.1 One MCP server, two clients

A `decipher-mcp` server (stdio transport) wrapping the investigation surface.
Client wiring:

- Claude Code: project `.mcp.json` entry.
- Codex CLI: `mcp_servers` entry in `~/.codex/config.toml`.

**Portability rule: tools-only.** MCP also defines resources, prompts,
sampling, and elicitation, but client support is uneven (Claude Code exposes
more than Codex historically). Everything we need expresses as tools with
JSON-schema inputs, which both clients handle identically.

### 3.2 Server-held state; sessions as stateless views

The server owns a persistent `InvestigationState` (the existing v3
serializable state, reused as-is). Client sessions hold **no** authoritative
state. This is v3's core design — "context rebuilt from state each turn;
loading the state *is* the resume path" — relocated into the server.

Consequences:

- A brand-new session (either harness) is fully briefed by one tool call
  (§4.3). No manual re-prompting, ever.
- An investigation started in Claude Code can be **continued in Codex** and
  vice versa — a capability the current harness cannot offer.
- Crash/restart recovery is the same one tool call.

Every mutating call carries an `expected_revision`; the server commits it
atomically and returns the new revision. A stale client receives a conflict and
must refresh. This is preferable to last-writer-wins because two valid clients
can otherwise silently fork the hypothesis board, repair ledger, or candidate
roles. Stdio server processes may be separate, but they coordinate through the
same locked on-disk investigation registry.

### 3.3 Epistemic controls and the migration rule

The gates that make results trustworthy are enforced in server code,
regardless of which client/model calls:

- `meta_declare_solution` refuses without a fresh, hash-matching, POSITIVE
  attestation (`reader_accepts_as_solution`), exactly per M5.3 Slice 6.
- Repair installation runs the Slice-4 host-validated acceptance checks
  (evidence binding, reject-before-install, default-deny on scalar decrease).
- Saturation counters and the `repair_exhausted` latch (Slice 2) apply.

MCP v1 begins at behavioral parity with the current host. Before Phase A, each
control is entered in a **policy provenance ledger** with:

- policy id and current enforcement point;
- original failure mode and motivating artifact/test;
- classification: invariant, evidence-management mechanism, or investigative
  policy;
- consequence if removed;
- proposed MCP form: hard enforcement, advisory recommendation, or telemetry;
- pre-registered evidence required to change that form.

The default classifications are:

1. **Invariants stay hard:** ground-truth isolation, cost ceilings, immutable
   provenance, hash freshness, transactional key consistency, unsupported-edit
   rejection, and declaration gating.
2. **Evidence mechanisms stay active:** hypothesis coverage, stale evidence,
   pending experiments, repeated-call detection, candidate diversity, and
   research history.
3. **Investigative policies are candidates for advisory mode, not immediate
   deletion:** scalar-selected workflow focus, phase/action restrictions,
   cipher-family sequencing, repair saturation policy, and verifier-to-route
   threshold rules.

Some existing checks straddle categories. In particular, default-deny repair
installation is an invariant, while "any scalar decrease is materially bad"
is a scoring-policy assumption. MCP v1 preserves the current behavior for a
clean comparison, but records the distinction so a weak scalar cannot become
an accidental permanent axiom.

### 3.4 The verifier-independence decision (key open design point)

v3's `verify` episodes are *fresh-context independent readers* whose verdicts
gate declaration. In the MCP design, who runs them?

- **Option A (recommended): server-side, API-billed.** The server runs verify
  episodes itself through the existing episode runner using a configured
  independent-reader tier (cents per verify in observed runs). Preserves the
  gate's independence property: the
  client agent cannot influence the reader's context. Small residual API
  spend, bounded and predictable.
- **Option B: client-side subagents.** Free under subscription, but the
  attestation is produced inside a context the lead agent shaped; the server
  can validate structure but not independence. Weakens the gate to
  self-attestation with paperwork.

Recommendation: **A**. The whole point of the gate is that the reader is not
the advocate. Keep experiments (no-LLM automated solvers) server-side too —
they cost nothing and already run as background compute.

Other reasoning work should *not* default to server-side API episodes in MCP
v1. The client agent should read candidates, compare partials, choose searches,
and formulate repair hypotheses itself. Otherwise the experiment would retain
the current API-billed cognitive loop behind an MCP facade and would not test
the architectural hypothesis. A future `delegate_worker` tool may be useful,
but its calls and spend must be explicit and separable from the client arm.

### 3.5 Tool surface (v1)

Small and lead-shaped (~15–25 tools), but not merely the current v3 lead
surface renamed:

- `investigation_start` / `investigation_status` / `investigation_list` —
  attach to a capability-scoped investigation, brief it, and enumerate only
  investigations visible to that capability (§4.3). Benchmark-backed
  investigations are provisioned by the trusted capsule launcher, not created
  by passing a benchmark id or filesystem path through MCP.
- `observe_overview`, `observe_diagnosis`, `decode_show` — compact reads.
- `hypothesis_branch_create` / `update` / `reject` / `next_steps` — board.
- `candidate_list` / `candidate_show` / `candidate_compare_signals` — a
  bounded, diverse portfolio with provenance and family-specific evidence;
  no single scalar silently defines the active candidate.
- `experiment_submit` / `experiment_collect` — background solver compute.
- `reading_record` / `comparison_record` — persist client-authored readings and
  rankings, hash-bound to the candidate content. A comparison records
  `best_partial` separately from `accepts_as_solution`.
- `repair_hypotheses_test` / `repair_transaction` — deterministic hypothesis
  compilation plus the validated install path.
- `request_independent_verification` — the one mandatory server-side reader
  operation (§3.4).
- `branch_adjudicate` — deterministic score/evidence comparison, not a hidden
  second planning agent.
- `meta_declare_solution` / `meta_declare_unsolved` — gated terminals.
- `act_set_model_variant` — language-model switching.

`next_steps` and similar outputs are explicitly advisory: they return policy
ids, uncovered refinements, and rationale, not a binding phase transition or
an allowed-action whitelist.

Deliberately NOT exposed in v1: the raw 95-tool v2 operator surface (the
frontier harness's own planning replaces most of it) and the generic
server-side `episode_run` cognitive loop. Expansion is easy later; shrinking a
published surface is not. Missing capabilities should be added as high-level,
evidence-producing tools rather than by exposing arbitrary Python or the
benchmark filesystem.

### 3.6 MCP and skills are complementary

- **MCP** supplies callable capabilities, scoped state, schemas, audit logs,
  concurrency control, and the security boundary. It is the portable core.
- **Skills/instruction files** teach each client how to investigate: preserve
  competing hypotheses, avoid repeated searches, interpret scores, test sparse
  nulls after distributed substitution damage, compare best partials, and
  verify before declaration. They are guidance and can be ignored or
  misunderstood, so they cannot enforce the firewall or epistemic gates.

The methodology should have one canonical versioned source. Thin Codex and
Claude adapters package that same source in each client's native format.

## 4. Prompting and session bootstrap

Three layers; none require re-typing anything.

### 4.1 Repo instruction files (standing methodology)

- A concise, canonical investigation-methodology document carries hypothesis
  discipline, coverage guidance, verify-before-declare, evidence-reading
  guidance, and the advisory descendants of the policy provenance ledger.
- Codex and Claude packages are generated or checked against that canonical
  source. `AGENTS.md` and `CLAUDE.md` may point to it for repository work, but
  the sanitized run directory receives only the relevant methodology — not the
  repository's large project-development instructions.

### 4.2 Invocable entry points

- Claude Code: a skill/slash command (e.g. `/investigate <file>`,
  `/resume-investigation`).
- Codex: custom prompts (`~/.codex/prompts/*.md`) providing the same entry
  points.

### 4.3 The self-briefing tool (the decisive layer)

`investigation_status(investigation_id)` returns the rebuilt factual brief:
cipher fingerprint and measured facts, a diverse candidate portfolio,
hypothesis board, coverage ledger, recent evidence, stale/fresh bindings,
pending experiments, research notes, budget, and state revision. A bare
session with zero prior context calls one tool and is mid-investigation.

Host recommendations are returned in a separately labeled block with policy
ids, provenance, and whether each is advisory or enforced. The status tool
must not smuggle the current scalar-selected workflow state and binding action
menu back into the new architecture as if they were neutral facts.

### 4.4 Sanitized run capsule (mandatory)

An exploratory or comparison run is created by a trusted launcher that writes
a **run capsule** containing only:

- ciphertext/transcription and its symbol/word structure;
- the explicitly permitted context tier;
- a random public case label that does not reveal family or solution;
- an opaque investigation capability/id;
- client methodology and writable notes/output directories.

It never contains plaintext, solution keys, grading alignments, benchmark
manifests, prior solved artifacts, or paths to them. The client process is
restricted to that directory (or a container with only that directory
mounted). The MCP server accepts only the opaque capability and never exposes
arbitrary path parameters. Ground truth is loaded later by the grading layer
after terminal submission; it is never mounted into or returned to the
model-facing client/server pair.

"Please do not inspect the benchmark" is not an acceptable firewall for the
§6 experiment. Doctrine is useful for ordinary research hygiene, but the
direction-setting comparison requires this technical boundary.

## 5. What stays on the custom harness (unchanged)

- **All scored benchmark evaluation**: firewalled, fixed-substrate,
  multi-model, multi-replicate (Track B runs, bake-offs, Stage-1 packets,
  Sequence-C-style acceptance). Any-model via API billing.
- Artifacts, the analyzer, Sequence-B replay, the M5.3 control machinery.
- Rationale: ordinary subscription sessions in the repository have filesystem
  access and vendor-tuned scaffolding, so their results are exploratory. Even
  sanitized-capsule comparisons answer an engineering question about the
  combined client+harness, not a clean model-science question.

## 6. The falsifiable experiment (decision gate)

Before implementing the server, complete **Phase 0: policy archaeology**. The
policy provenance ledger must cover at least context-family discipline,
verification gating, repeated-call suppression, repair acceptance,
saturation, workflow routing, and scalar candidate focus. MCP v1 then retains
the hard controls and exposes the investigative policies as visible advice and
telemetry. This avoids declaring victory by silently removing safeguards that
were introduced after real failures.

Before investing past v1, run the comparison the architecture question
actually turns on. Every model-visible arm receives the same sanitized case
payload and allowed context; only the post-run grading layer sees ground truth:

- **Arms**: (1) Claude Code + decipher-mcp (Claude, Max); (2) Codex +
  decipher-mcp (gpt-5.x, Pro); (3) v3 harness (gpt-5.5, API) as baseline.
- **Cases** (the agent-critical set): `borg_single_B_borg_0109v` (the fixed v3
  run reached 91.0/66.7 unsolved at $1.43),
  `copiale_single_B_copiale_p017` (54.6 v2 vs 75.4 v3), and the Quagmire-3
  synthetic (`synth_en_97q3nb_s50`). Add at least two newly generated,
  opaque-id held-out analogs: one sparse-null substitution and one
  wrong-family lure. Famous or repository-visible cases alone are not adequate
  evidence. Preflight and local-solver access must be identical across arms;
  either all arms receive the same initial candidate or all must invoke the
  same MCP/local experiment themselves. Do not compare a no-preflight arm to
  the cited preflight-assisted baseline.
- **Metrics**: char/word accuracy, API dollars, subscription capacity where
  observable, tool/model calls, wall clock, strongest post-hoc candidate ever
  generated versus retained, family/refinement coverage, and qualitative
  decision review from artifacts/transcripts. Record attempted policy
  violations and server interventions (blocked declarations, duplicate calls,
  unsupported edits), not only terminal compliance — a perfect final gate
  score can conceal a badly disciplined client repeatedly hitting barriers.
- **Reading the result**: if a frontier harness + tools matches or beats v3
  on these cases, the custom loop shrinks to an eval shell and future loop
  investment (e.g. parts of M5.4) is redirected to the MCP surface. If v3
  wins clearly, the harness has earned its keep with data.
- **Caveat recorded up front**: cross-harness arms confound model with
  harness; this experiment picks an *engineering direction*, it is not model
  science. GT-scored, but treated as exploration-grade evidence.
- **Ambiguity rule**: if the external arms and v3 are close, do not infer that
  either architecture won. Add a thin MCP client driven through the API by the
  same model/configuration as v3 to isolate the effect of policy and tool
  surface from the effect of the vendor harness.
- **Pre-registration**: define pass/fail thresholds before paid runs, including
  maximum false/ungated declaration attempts, maximum repeated expensive
  actions, minimum candidate-retention rate, and a bounded spend/capacity
  envelope. Do local replay and synthetic checks first; do not launch a broad
  bake-off to answer an architectural prototype question.
- **Staging and replication**: first run one pilot replicate on the two new
  opaque analogs plus one historical case. If the MCP path is operationally
  credible, run at least two confirmation replicates per arm on the smallest
  case subset that discriminated between policies. A one-replicate result is a
  debugging observation, not evidence for decommissioning v3 or retiring a
  host-side control.

## 7. Modularity, refactoring cost, and the no-breakage rule

**Hard rule (user, 2026-07-17): the current v3 agent must not break at all.
No code duplication; the server and v3 share one implementation.**

### 7.0 What the code actually requires (seam analysis)

Reusable AS-IS, zero refactor, zero duplication — these are already
loop-independent modules the MCP server imports directly:

- `InvestigationState` + serialization (`investigation/state.py`) — the
  shared substrate; the revision-checked store wraps it.
- `context.py` (`workflow_state`, `build_lead_context`) — becomes the body of
  `investigation_status` (rendered to one text blob).
- `WorkspaceToolExecutor` + `AttestationPolicy` (`agent/tools_v2.py`) —
  already constructed standalone with injected policy.
- `episodes.run_episode` — already callable outside the loop (the lead
  dispatches it); backs `request_independent_verification`.
- The experiment queue (`experiments.py`) — dispatchers already take
  `(queue, state, workspace, executor, args, turn)`.
- Composites (`actions.execute_composite`), `model_provider`/`sessions`,
  scorers, menus, solvers.

**The one real refactor**: `run_v3`'s lead *dispatch layer* — the nested
closures `_dispatch_episode_run` / `_dispatch_episode_install` /
`_dispatch_repair_transaction` (+ `_settle_repair_outcome`, saturation
bookkeeping, duplicate suppression, episode-kind gating), roughly 600–800
lines entangled with the turn loop's `emit` and bookkeeping. Extract them
into a loop-independent host object (working name `InvestigationHost`)
owning `(state, workspace, executor, queue, emit)` and exposing
`handle_tool(name, args) -> result`. After extraction:

- `run_v3` becomes a thin driver: build context → `session.send` → for each
  tool_use → `host.handle_tool` → bookkeeping. Same behavior, same events,
  same artifacts.
- The MCP server is a second thin driver over the *same* host (plus the
  revision store and capsule layer). One implementation, two entry points.

### 7.1 Why v3 cannot break

1. The extraction lands FIRST as its own zero-behavior-change slice through
   the standard pipeline (spec → coder → adversarial review → land), before
   any MCP code exists.
2. The entire existing suite (~1,690 tests) exercises the dispatch layer
   THROUGH `run_v3` with scripted sessions — M5.3's saturation, acceptance,
   and gate tests all drive the exact code being moved, so a
   behavior-preservation failure is caught by construction, not by new tests.
3. The Sequence-B replay test pins the end-to-end lead path.
4. The MCP server is a new module + new entry point that *imports* the host;
   the v3 CLI path never imports MCP code. Addition cannot break it.
5. Optional belt-and-suspenders: one ~$1.3 paid parity smoke
   (borg_0109v, v3) after the extraction lands, before Phase A.

### 7.2 Cost estimate

- Host extraction slice: comparable to one M5.3 slice (days, not weeks, at
  the current cadence).
- MCP server adapter: ~300–500 lines of new adapter/capsule/revision code
  (Phases A–B), everything else reuse.
- Client integration + doctrine files: small (Phase C).

## 7½. Phased plan (sizes are rough; no code in this proposal)

- **Phase 0 — policy archaeology + acceptance design** (small): build the
  provenance ledger; classify controls; define sanitized-capsule requirements,
  experiment thresholds, and which v3 behavior MCP v1 must preserve.
- **Phase A — server skeleton + sandbox launcher** (small/medium): MCP server
  process, capability-scoped investigation registry, atomic revision/locking,
  `investigation_start/status`, sanitized run-capsule creation, and state
  persistence on disk. Reuse `InvestigationState`; reuse factual context
  renderers selectively rather than importing the binding workflow menu
  verbatim.
- **Phase B — surface + gates** (medium): the §3.5 tool set delegating to
  existing executors/dispatchers; Slice-2/4/6 gate enforcement server-side;
  server-side verify episodes (Option A), the experiment queue, candidate
  portfolios, client-authored evidence records, and deterministic repair
  compilation. No general server-side cognitive `episode_run` in v1.
- **Phase C — client integration** (small): `.mcp.json` + Codex config,
  canonical methodology, thin client-specific skill/instruction adapters, and
  capability-scoped run directories.
- **Phase D — the §6 experiment** (mostly run-time + review): the staged pilot,
  then the minimum discriminating confirmation packet; write-up;
  policy-intervention audit; direction decision.

Dependency note: CLI-3 (human-first display) is in flight and unaffected —
it improves the eval harness we are keeping regardless.

## 8. Risks and open questions

1. **Verifier independence** (§3.4) — recommended resolution: server-side,
   small API spend. This is the one place "fully subscription-funded" is
   deliberately traded for integrity.
2. **Concurrent sessions** on one investigation (Claude Code and Codex
   simultaneously): v1 uses a per-investigation write lock plus optimistic
   revisions. Do not use last-writer-wins for evidence state.
3. **MCP client feature drift**: mitigated by the tools-only rule.
4. **Subscription capacity limits**: exploration fits; anything matrix-shaped
   must go through the API harness. Set that expectation in AGENTS.md so a
   session doesn't burn a weekly cap on a sweep.
5. **Benchmark leakage in exploratory sessions**: ordinary repository sessions
   can read benchmark data and remain exploration-only. The §6 comparison is
   different: its client runs in a sanitized capsule with no benchmark mount,
   and its MCP capability cannot request target solution data. Treat sandbox
   escape or server path traversal as a failed run, not a review caveat.
6. **Model-comparison hygiene**: nothing from §6 or subscription sessions
   feeds the M6/default-loop decision tables; those remain API-harness-only.
7. **Policy amnesia**: translating a hard v3 behavior into a skill can make a
   proven safeguard optional by accident. Mitigation: policy ids appear in
   recommendations and telemetry; changes require the Phase-0 retirement
   criterion and a targeted ablation.
8. **MCP output/context inflation**: a state dump can recreate transcript
   bloat. Mitigation: paginated candidate/evidence tools, compact status
   packets, stable ids, and explicit detail calls.
9. **Subscription economics are capacity economics, not zero cost**: provider
   limits and model availability can change. Record calls/tokens/capacity when
   available and keep matrix evaluation on the API harness.
10. **Stochastic conclusions from tiny samples**: a strong or weak single run
    can reflect model variance rather than architecture. Use the staged
    replication rule in §6, and do not retire controls based on an anecdotal
    win.

## 9. Relationship to the existing roadmap

- M5.3 is complete and unaffected.
- **M5.4 (repair reframe / interpretation packets):** host-independent data
  models, deterministic compilers, and earned-override invariants remain useful
  under either driver and need not be discarded. Defer the lead-loop-specific
  interaction workflow until §6 determines whether the primary driver is our
  loop or a frontier harness over MCP tools.
- INV (investigator mode) is the natural first *consumer* of the MCP surface:
  real unsolved ciphers, no firewall requirement, long exploratory sessions —
  exactly the subscription-economics sweet spot.

## 10. Initial policy-provenance seed

Phase 0 should expand this table with exact artifact/test references before
implementation:

| Policy | Why it exists | Initial class | MCP v1 form |
|---|---|---|---|
| Ground-truth firewall | Solver workflow previously consumed grading data | Invariant | Hard server/capsule boundary |
| Fresh positive verification before solved declaration | Agents declared readable or memorized-looking junk | Invariant | Hard, hash-bound server gate |
| Transactional repair evidence binding | Readers proposed prose or unsupported mappings as edits | Invariant | Hard compiler/install validation |
| Duplicate expensive-call suppression | Agents repeated unchanged searches and reads | Evidence + resource invariant | Hard exact-duplicate block; telemetry for near-duplicates |
| Context-family prior and override rationale | Agents wandered into unrelated families despite explicit context | Investigative policy with budget consequences | Strong advisory + coverage telemetry initially; retain a bounded-spend guard until ablated |
| Scalar-selected workflow focus | Needed a deterministic branch when agents failed to choose | Investigative policy | Diverse portfolio; scalar is one signal only |
| Verifier-to-repair/broaden thresholds | Agents polished basin-wide gibberish or abandoned useful local damage | Investigative policy | Advisory route with policy id; record client choice and outcome |
| Repair saturation latch | Agents retried ineffective repairs indefinitely | Mixed evidence/resource policy | Preserve in v1; distinguish exact repeated evidence from a genuinely new repair hypothesis |
| Compare `winner` requirement | Needed a hash-bound terminal fallback | Contract policy | Split `best_partial` from `accepts_as_solution`; bind both |
