# Agent Loop v3 M5.2 - Strategy and Repair Reliability

Status: implementation complete. Slices 1-5 implemented 2026-07-16. Paid acceptance remains
deferred until the user approves another provider spend.

## Motivation

The M6 bake-off and M5.1 Stage-1 runs showed that the v3 architecture is
cheaper and broadly competitive, but its implementation still let the lead
behave like a lower-cost v2 operator. The lead received the complete v2 tool
library, repeated cheap reads and low-level edits, and did not reliably turn a
negative verification into a bounded repair transaction. Three reading
episodes exhausted their budgets without submitting, and real worker outputs
used ambiguous numeric positions that the host interpreted as token offsets.

This milestone finishes the strategist/operator separation and makes reading
and repair contracts host-verifiable. It must remain ground-truth-free at
runtime.

## Slice 1 - Strategic Lead and Stable Reading Packets

Implemented in this change:

1. The ordinary lead surface contains 16 tools including experiment tools,
   rather than the former 102-tool effective surface. Benchmark-context runs
   add six scoped context readers for 22 total. Observation, search, key-edit,
   and reading-application primitives remain available to scoped workers.
2. The ordinary serialized tool schema is about 10.9k characters (roughly
   2.7k tokens), down from about 66k characters before experiment tools.
3. The rebuilt lead view includes an explicit workflow state and a short menu
   of valid high-level actions. A repairable negative attestation produces
   `repair_required`; a collapsed reading produces `broaden_required`; a fresh
   positive attestation produces `verified`.
4. Reading episodes receive a host-generated `CandidateReadingPacket` with the
   exact candidate content hash, candidate capability, and opaque span ids.
   Workers reference span ids; the host resolves them to token ranges and
   ignores worker-authored numeric offsets when a packet is present.
5. Metadata-only candidates are explicitly marked `text_only`, preventing a
   reading from being mistaken for a key-editable candidate.
6. Reading work is bounded to four tool calls because the full candidate is in
   the packet. On exhaustion, the final send exposes only
   `episode_submit_result`; it no longer removes tools and asks for raw JSON.

Local acceptance:

- focused v3 context/episode/reading/loop tests;
- attestation, ground-truth firewall, hypothesis-action, state, session, and
  experiment tests;
- schema-size regression and low-level-tool absence assertions;
- opaque-span regression where bogus worker numeric offsets cannot move the
  repair range.

## Slice 2 - Enforced Workflow Transitions

Implemented 2026-07-16.

The lead's visible tools are bounded and the dispatcher now enforces that
surface. Historical exchanges remain context only and do not restore legacy
tool authority. The transition policy now:

- derives allowed episode kinds from `searching`, `candidate_reading`,
  `repair_required`, `broaden_required`, and `verified`;
- narrows the `episode_run.kind` enum each turn;
- rejects hidden direct tools with a structured `lead_tool_not_available`
  result;
- preserves already-paired historical exchanges without granting an escape
  hatch; and
- suppresses duplicate reads keyed by tool name, normalized arguments, and
  unchanged branch content hash.

## Slice 3 - Repair-Capable Candidate Representation

Implemented 2026-07-16.

Branch-shape special cases are now normalized through one candidate contract
carrying:

- exact rendered text and content hash;
- provenance (solver, finalist, episode/experiment);
- editable key, null mask, boundary overlay, and transform pipeline when
  present;
- declared capabilities: `editable_key`, `editable_null_mask`,
  `editable_boundaries`, `editable_transform`, or `text_only`;
- token-to-rendered-text provenance sufficient for aligned repairs.

Null-mask finalists must retain their underlying homophonic key and mask rather
than installing only `metadata.decoded_text`. This is required before Copiale
can be a meaningful repair acceptance case.

Installed null-mask finalists now declare their renderer and repair
capabilities, while the host dynamically renders their retained key with the
selected mask. Reading spans carry the filtered effective-token indices, so a
repair against compressed display text maps back to the correct cipher
tokens. Tool panels and verify attestations use the same renderer and content
hash. Metadata-only overlays remain explicitly `text_only`.

## Slice 4 - Transactional Repair

Implemented 2026-07-16.

One high-level repair transaction consumes a candidate packet plus a
Reading or verify anomalies and returns either an improved installed snapshot
or a structured failure:

1. generate conservative variants on isolated forks;
2. score collateral damage and reject unsupported edits;
3. keep a small diverse finalist set;
4. compare the finalists;
5. install the supported winner;
6. require re-verification for changed content.

The transaction records which anomalies it addressed. A negative verification
hint must not recur for the same content after the transaction has handled it.

The transaction binds a stored Reading to the source branch's exact content
hash, injects the candidate packet and latest verifier anomalies into one
isolated repair episode, rejects stale readings, unchanged results, ambiguous
finalists, and unsupported claimed winners, then installs one changed result.
It records source/result hashes and addressed anomalies durably. The installed
candidate becomes the active workflow branch until fresh verification, so the
lead cannot immediately fall back into the handled source hint.

## Slice 5 - Durable State and Honest Termination

Implemented 2026-07-16.

- Content-identical episode installs deduplicate onto the existing branch and
  preserve a durable provenance alias.
- Survey episodes add structured evidence and update the hypothesis board;
  episode and experiment installs inherit board state with their result as the
  evidence source.
- Negative verifier anomalies create durable repair-agenda items, and a
  successful bound repair transaction closes the source-content items it
  addressed.
- Content-bound call signatures and no-new-information streaks are tracked and
  survive resume.
- Negative-only exhaustion now retains a best-effort selection but terminates
  honestly `unsolved`; provider failures remain `error`. Only a fresh positive
  attestation may synthesize `fallback_declared`.

## Verification Plan

No full bake-off is required between implementation slices.

1. Run local unit and replay tests after every slice.
2. Replay stored M6 and M5.1 reading packets without provider calls.
3. With explicit user approval, run one real-model reading submission smoke on
   a stored candidate and cap its cost tightly.
4. Then run targeted Borg 0109v replicates, aiming for verification by turn 3
   and recovery toward the v2 96% character / 75% word basin.
5. Run Borg 0045v targeted replicates, requiring early null-mask portfolio use
   and at least the prior 80% character floor.
6. Run Copiale p017 only after null-mask candidates are repair-capable.
7. Do not run the paired M6 bake-off or switch the default until the user
   approves the spend and the targeted gates pass.

## Non-Goals

- No ground-truth-guided routing, selection, repair, or retry behavior.
- No Borg- or Copiale-specific prompt rules.
- No budget increase as a substitute for a bounded worker contract.
- No default-loop switch in this milestone.
