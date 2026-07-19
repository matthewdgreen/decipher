# Spec — client-reading verification tier (keyless graceful degradation)

Status: DRAFT for implementation. Author: Fable (main loop), 2026-07-19.
Depends on: investigation CLI I-0 (the operation manifest — this feature adds
one `OperationSpec`). Sequence AFTER I-0 lands.

## 1. Why

A user with no API key hits a wall: `request_independent_verification` returns
`no_verification_provider`, and because `meta_declare_solution` is hard-gated on
a positive attestation (DECL-1), a keyless run can produce a correct decode but
can never reach a *declared* solved state — it dead-ends at "promising but not
independently verified."

The obvious fix — borrow the client's own model via MCP **sampling** — is off
the table: sampling is unsupported in both Claude Code
([anthropics/claude-code#1785](https://github.com/anthropics/claude-code/issues/1785))
and Codex ([openai/codex#4929](https://github.com/openai/codex/issues/4929)),
AND it is **deprecated** as of MCP spec 2026-07-28 (SEP-2577), with explicit
guidance to migrate to direct provider APIs. So the reverse-channel is closed.

This spec instead adds a **client-initiated** self-read tier: the driving agent
(which already has a model) reads the candidate itself — ideally via a
fresh-context sub-agent — and records the verdict through a new tool. No
server→client channel needed; works on both clients today.

Owner decision (2026-07-19): adopt option (B), client self-read. Deliberately
NOT option (A) local Ollama reader — its time/memory cost is too high for most
users and small-model quality is unproven (that quality question is a separate
TODO, see [[local-reader-quality-todo]]).

## 2. Binding invariants (what this must NOT erode)

- **The gate stays real where it counts.** Server-side independent verification
  (GT-2) is unchanged and remains the ONLY tier that counts for scored/eval
  work. The benchmark/parity harness never uses the client tier.
- **Content-hash discipline (DECL-1) is preserved** for the client tier too: a
  client attestation is bound to the branch's current rendered decode hash;
  A-verified-then-mutated-to-A′ laundering is blocked regardless of who read it.
- **Honest labeling, never masquerade.** Every attestation carries a
  `verifier_tier`; a client-tier declaration is visibly *provisional /
  self-verified*, distinct from independently-verified, in the terminal record,
  the status brief, and the decode the agent shows the human.
- **Firewall (GT-1..3) untouched.** The client reader receives ONLY the
  candidate decode + target language (mirroring the server reader's GT-2
  isolation); no ground truth, no scores. This is CLIENT-enforced and therefore
  best-effort/advisory — which is precisely why the tier is weaker and
  eval-rejected.

## 3. The mechanism

### 3.1 `verifier_tier` on every attestation
Add `verifier_tier: "server_independent" | "client_self_read"` to the
attestation record (default `server_independent` for the existing server path —
no behavior change to it). One shared predicate reads it; nothing else about the
attestation schema changes.

### 3.2 The refusal becomes the instruction
When `request_independent_verification` resolves keyless (today's
`no_verification_provider`), return an ACTIONABLE payload instead of a dead end:
```
{ "reason": "no_verification_provider",
  "client_reading_available": true,
  "instructions": "No server-side verifier is configured. You MAY perform a
     client-side reading: spawn a FRESH-CONTEXT sub-agent (a separate context,
     optionally a different model), give it ONLY the branch decode from
     decode_show and the target language — NOT your solving history or scores —
     have it judge readability against the rubric, then record the verdict with
     attest_client_reading at this revision. This is a WEAKER, self-verified
     tier: it will unlock a PROVISIONAL declaration, not independent
     verification, and does not count for benchmarking.",
  "rubric": { "reader_accepts_as_solution": "bool",
              "target_language_confidence": "0..1",
              "semantic_recoverability": "0..1",
              "coherence": "0..10",
              "damage_scope": "local|distributed|basin_wide" },
  "candidate_ref": {"tool": "decode_show", "branch": "<branch>"} }
```
The agent reads tool results attentively, so this in-band instruction is the
reliable channel; onboarding §2 / AGENTS.md carry a backup description.

### 3.3 New tool `attest_client_reading` (mutate)
- Input: `investigation_id`, `expected_revision`, `branch`, and the rubric
  scalars (the sub-agent's verdict).
- The dispatcher computes the branch's current content hash at dispatch (same as
  the server verify path — DECL-2 single-writer discipline) and writes an
  attestation with `verifier_tier: "client_self_read"`, hash-bound.
- The server does NOT run a model; it records the client-supplied verdict. The
  independence is the client's responsibility (hence the tier).
- Conservative coercion identical to the server path (coherence clamps,
  unit-interval clamps, non-positive defaults) so a scale-violating client can't
  mint a spurious positive.
- Registered as one `OperationSpec` in the manifest (post-I-0) → reaches both
  the MCP surface and the future `decipher investigation` CLI automatically.
  `external_effect: NEVER` (no server-side external call).

### 3.4 Declaration gate branch (DECL-1)
`meta_declare_solution` currently requires the newest hash-matching attestation
to be positive. Extend, do not weaken:
- A positive `server_independent` attestation → declared `solved` (unchanged).
- A positive `client_self_read` attestation, AND no server verifier is
  configured on this server → declared `solved_provisional` (a distinct terminal
  marker, or `solved` + `verification_tier: client_self_read` — implementer
  picks the least-schema-churn option; the REQUIREMENT is that the tier is
  unmistakable downstream).
- If a server verifier IS configured, a `client_self_read` attestation does NOT
  unlock declaration — the client should use the real reader. (Don't let a lazy
  client route around a present server reader.)

### 3.5 Eval / benchmark rejection (ENFORCED)
- The benchmark/parity harness configures a real verify provider and its
  declaration path accepts ONLY `server_independent`. It never surfaces
  `attest_client_reading`.
- `scripts/grade_dual_harness_run.py` records `verifier_tier` on graded rows and
  EXCLUDES `client_self_read` declarations from any "verified" aggregate — a
  provisional self-read is reported as such, never as an independent verify.
- Provenance ledger: add the client tier as an **advisory, MCP-surface-only**
  control with an explicit note that it is hard-rejected by the scored path
  (a DECL-1 sibling; class POL/advisory, never INV).

## 4. Slices

- **CR-1**: `verifier_tier` field + the shared predicate; server path defaults to
  `server_independent` (byte-parity — no behavior change). Tests.
- **CR-2**: `attest_client_reading` tool (manifest OperationSpec + service
  dispatch + hash-binding + coercion). Keyless `request_independent_verification`
  returns the actionable payload. Tests.
- **CR-3**: DECL-1 branch for the client tier (provisional declaration; blocked
  when a server verifier exists). Terminal record + status brief + the
  "Show the text" surface all label the tier. Tests.
- **CR-4**: eval rejection — grader tier column + exclusion; benchmark path
  refuses client tier; ledger note; onboarding §2 + AGENTS.md doctrine
  ("keyless? you may self-read via a fresh sub-agent; it's provisional").

## 5. Acceptance

- **Keyless provisional close**: on a server with `--verify-provider none`, a
  correct decode → `request_independent_verification` returns the actionable
  payload → the agent records a positive `attest_client_reading` →
  `meta_declare_solution` succeeds as `solved_provisional` with
  `verifier_tier: client_self_read` visible.
- **Server-reader unchanged**: with a provider configured, the existing
  server-independent verify→declare path is byte-identical, and a
  `client_self_read` attestation does NOT unlock declaration.
- **Eval integrity**: a `client_self_read` declaration is excluded from graded
  "verified" aggregates and refused by the benchmark declaration path.
- **Hash discipline**: mutating the branch after a client attestation invalidates
  it (stale hash) exactly as for the server tier.

## 6. Orchestration

Per CLAUDE.md: this spec (Fable) → per-slice Opus/Sonnet coders → Fable review →
commit per slice. Sequence AFTER investigation CLI I-0 (the manifest). CR-2's
tool is one manifest `OperationSpec`, so it lands dual-surface for free — and
`tests/test_interface_parity.py` (from CLI I-1) will cover the operation. This
is deliberately a SMALL program (one field, one tool, one gate branch, one
grader change); it is not a redesign of verification.
