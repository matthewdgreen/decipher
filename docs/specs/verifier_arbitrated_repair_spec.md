# Verifier-Arbitrated Repair Acceptance — Implementation Spec

Status: ready to implement. Written 2026-07-17 against HEAD `0dfef8a` (all line
numbers below are anchored there; re-locate by the quoted code if the tree has
moved). Baseline: full suite **1753 passed / 2 skipped** at `0dfef8a`.

Authority: this spec. It **supersedes** the truncated, untracked draft
`docs/specs/mcp_verifier_arbitrated_repair_spec.md` (26 lines, cut off
mid-sentence); do not implement from that file. One deliberate divergence from
that draft: the v3 host tool DOES expose the opt-in flag too (same flag, same
default-false), because default-false already guarantees byte-identical v3
behavior — there is no need to make the MCP server the only caller.

Pre-registered evidence (read it first):
`docs/evidence/c56de7e6c600_repair_guard_false_reject.json` — two real
Codex-MCP cases, summarized in §1.

Related contracts: M5.3 Slice 4 acceptance
(`docs/specs/agent_v3_m5_3_control_reliability_spec.md` "Slice 4 -
Host-Validated Repair Acceptance"), provenance ledger rows REP-3/REP-4
(`docs/mcp_policy_provenance_ledger.md` §3), MCP server spec
(`docs/specs/mcp_server_spec.md`).

---

## 1. Motivation and evidence

### Case 1 — `c56de7e6c600`: the false reject this spec fixes

A real Codex MCP session cracked a homophonic cipher. Its
`repair_transaction` (transaction `972ac7027c64`, mode `client_compiled`)
carried a batch of objectively correct symbol corrections (among them
`TRAILER→TRAWLER`, edits `S013:T->C`, `S023:I->W`). It passed acceptance
checks 1–5 (winner_named, worker_applied, winner_fork_evidence,
edit_claims_bound, winner_adjudicated) and FAILED check 6
`collateral_within_limits` → `materially_non_improving`, because the
deterministic occurrence counter judged against the 5,000-word common list:
TRAWLER is not in the list, TRAILER is (`damaged_occurrences: 3,
improved_occurrences: 0`). The scalar probe never ran (`scores_after: null`,
`score_deltas: null` — check 6 fires before checks 7/8). Meanwhile an
independent verify reader in the same investigation scored the corrected
reading `target_language_confidence 0.97`, `semantic_recoverability 0.88`.
The strongest evidence source we have — an independent reader — was silently
overruled by a counting heuristic.

This is exactly the §3.3 "straddle" the MCP proposal pre-registered and the
ledger records at REP-4: **default-deny INSTALL is an invariant, but "any
mechanical decrease is materially bad" is a SCORING POLICY, not an
invariant.**

### Case 2 — `776221457325`: the related routing failure (scope note only)

A third real Codex run (no-boundary homophonic, 98.2% char) shows a
related-but-distinct persistence failure: the reader identified 6 correct
single-symbol key fixes (`HOLLOWED→FOLLOWED`, `DUST→DUSK`, `THANT→THANK`,
`DEAR→YEAR`, …) but **no `repair_transaction` was attempted at all** — the
verify verdict was `damage_scope="distributed"` (6 scattered errors), which
routes away from local repair toward broaden, even though each error is an
individually simple, batch-repairable key edit.

Same root problem ("a correct reading cannot be persisted"), different
mechanism. The CORE fix in this spec is verifier-arbitrated ACCEPTANCE
(case 1). Case 2 gets two light touches only (§8): a test confirming the
acceptance/arbitration path works for a batch repair fixing scattered errors
in one fork, and a doctrine line. **No routing/gate redesign.** Note the
damage-scope→route mapping is ledger row **WF-4**, already **advisory** in
the MCP surface — so the doctrine line is guidance, not a gate change.

---

## 2. Design summary

When the two SCORING checks inside `validate_and_install_repair` — check 6
`collateral_within_limits` (host.py:1515–1526) and check 8
`scalar_non_decrease` (host.py:1555–1570) — would REJECT a repair, an
**opt-in** arbitration path runs ONE fresh, server-side, independent verify
episode on the repaired fork's content (the same independent-reader machinery
as `_dispatch_verify_run`: empty toolset, candidate text + language only, no
lead influence), hash-bound to the post-repair content. If the independent
reader judges the repaired fork **strictly better** than the incumbent (§4's
concrete rule), the repair INSTALLS, with the arbitration decision and both
attestations recorded in the transaction's acceptance sub-record. Otherwise
it REJECTS exactly as today.

Mental model: *arbitration is the verify you would have run after install,
run before install instead, and its verdict decides the install.* On accept,
the attestation it produced is written to `state.verify_attestations` against
the installed branch — identical in trust terms to calling
`request_independent_verification` immediately after the install.

## 3. Hard invariants (state these in code comments; test them)

1. **Reader independence, Option A.** The arbitrating reader is the SAME
   server-side verify episode (`EpisodeSpec(kind="verify")`, empty toolset,
   inputs = candidate text + language only). Never client-attested; no
   client- or lead-authored text reaches it (the `goal` is `""` and the
   verify context builder ignores goals anyway, GT-2).
2. **Nothing installs on the model's say-so.** Arbitration is a SERVER-run
   verify, hash-bound: the candidate text is rendered host-side from the
   winner snapshot with the pinned canonical renderer (CMP-3), and its
   sha256 must equal the snapshot digest already recorded in `changed`.
3. **Mechanical checks remain the DEFAULT.** Arbitration is opt-in per call
   (new boolean arg `verifier_arbitration`, default false) and ENGAGES only
   when check 6 and/or check 8 would otherwise reject. It never engages for
   checks 1–5 (evidence binding) or check 7 (`no_op_probe`) — those are
   invariants (REP-1/REP-3) and are never arbitrable.
4. **It never LOOSENS a real regression.** A genuinely worse repair still
   fails because the independent reader will not prefer it (§4 rule); an
   arbitration-rejected transaction fails with the same
   `materially_non_improving` reason, same `evidence` failure class, same
   saturation accounting as today.
5. **v3 internal repair path behavior UNCHANGED unless the flag is passed.**
   Default false = today's behavior byte-for-byte, including the acceptance
   sub-record (no new keys, `policy` string unchanged). All existing repair
   tests pass unmodified.
6. **Keyless degradation.** No verify capability available → arbitration
   unavailable → falls back to today's mechanical reject with a typed
   `no_verification_provider` (or `cost_ceiling_reached`) reason inside the
   acceptance sub-record. The top-level failure reason stays
   `materially_non_improving` so failure classification and saturation are
   unchanged.

Additional preserved properties (do not regress):

- `REPAIR_ACCEPTANCE_POLICY` stays `None` and the `assert
  REPAIR_ACCEPTANCE_POLICY is None` at host.py:1563 stays. Arbitration is
  NOT the M5.4 worker-improvisable policy object; it is a new independent
  EVIDENCE source, host-run.
- DECL-2 single-writer: only the dispatcher/host writes attestations.
- SAT-1/SAT-3/REP-2: a rejected arbitration consumes evidence-failure budget
  exactly like today's `materially_non_improving`, and the
  (`source_content_hash`, `interpretation_digest`) pair is then blocked by
  `pair_evidence_failed` — so at most ONE arbitration verify can ever run
  per (content, interpretation) pair. This is the cost-containment story.
- BUD-1: the arbitration verify is a paid episode; the host checks
  `cost_ceiling_reached()` before dispatch and passes
  `max_cost_usd`/`outer_cost_usd` into `run_episode` as
  `_dispatch_verify_run` does.

## 4. The pre-registered decision rule ("strictly better")

New module constants in `src/investigation/host.py`, directly below
`REPAIR_ACCEPTANCE_POLICY` (line 208):

```python
# Verifier-arbitrated repair acceptance (docs/specs/verifier_arbitrated_repair_spec.md).
# Pre-registered constants; motivating evidence:
# docs/evidence/c56de7e6c600_repair_guard_false_reject.json. Changing any of
# these requires new ledger-recorded evidence (REP-4 row).
ARBITRATION_POLICY_ID = "verifier_arbitration_v1"
ARBITRATION_MARGIN = 0.05
ARBITRATION_FLOOR_TLC = 0.90
ARBITRATION_FLOOR_RECOVERABILITY = 0.60
```

New pure module-level function (unit-testable in isolation), placed next to
the other module helpers (e.g. after `_winner_adjudication_summary`):

```python
def _arbitration_verdict(
    repaired: dict[str, Any], incumbent: dict[str, Any] | None
) -> tuple[bool, str | None, dict[str, Any]]:
    """Decide whether the independent reader prefers the repaired fork.

    ``repaired`` is the arbitration AttestationRecord as a dict (fields
    already clamped by the record constructor). ``incumbent`` is the newest
    stored attestation for the PRE-repair source content, or None.
    Returns (accepted, rule, detail); ``rule`` is None when rejected.

    Rules, in order:
    1. reader_accepts_as_solution — the reader accepts the repaired text
       outright as a solution (the DECL-1 positive predicate).
    2. margin_improvement — an incumbent with scalar fields exists; the
       repaired reading must not decrease EITHER reader scalar and must gain
       at least ARBITRATION_MARGIN combined.
    3. absolute_floor — no usable incumbent baseline (none stored, or a
       legacy record without target_language_confidence): with nothing to
       compare against, only a strong absolute verdict can overrule the
       mechanical reject. Floors are deliberately stricter than the WF-4
       "high" routing thresholds (0.7 / 0.5, context.py).
    """
    r_tlc = clamp_unit_interval(repaired.get("target_language_confidence"))
    r_rec = clamp_unit_interval(repaired.get("semantic_recoverability"))
    if repaired.get("reader_accepts_as_solution") is True:
        return True, "reader_accepts_as_solution", {
            "repaired_tlc": r_tlc, "repaired_recoverability": r_rec,
        }
    if incumbent is not None and "target_language_confidence" in incumbent:
        i_tlc = clamp_unit_interval(incumbent.get("target_language_confidence"))
        i_rec = clamp_unit_interval(incumbent.get("semantic_recoverability"))
        d_tlc, d_rec = r_tlc - i_tlc, r_rec - i_rec
        detail = {
            "repaired_tlc": r_tlc, "repaired_recoverability": r_rec,
            "incumbent_tlc": i_tlc, "incumbent_recoverability": i_rec,
            "delta_tlc": round(d_tlc, 6), "delta_recoverability": round(d_rec, 6),
        }
        accepted = (
            d_tlc >= 0.0 and d_rec >= 0.0
            and (d_tlc + d_rec) >= ARBITRATION_MARGIN
        )
        return accepted, ("margin_improvement" if accepted else None), detail
    detail = {"repaired_tlc": r_tlc, "repaired_recoverability": r_rec,
              "incumbent": None}
    accepted = (
        r_tlc >= ARBITRATION_FLOOR_TLC
        and r_rec >= ARBITRATION_FLOOR_RECOVERABILITY
    )
    return accepted, ("absolute_floor" if accepted else None), detail
```

Sanity anchors: motivating case 1 had NO incumbent attestation
(`attestation_key: "none"`) and a repaired reading of 0.97/0.88 → rule 3
accepts. A repaired reading the reader scores below the incumbent on either
scalar → rejected. A reader that omits the scalar fields yields 0.0/0.0 after
clamping → rejected unless rule 1 fires (conservative, matching DECL-4's
philosophy). A legacy incumbent record lacking the
`target_language_confidence` key is treated as ABSENT (rule 3), so a 0.0
legacy baseline can never make a weak repaired reading look like a margin
improvement.

## 5. Exact edits

### 5.1 `src/investigation/sessions.py` — availability helper

Add next to `register_session_builder` (line 418):

```python
def has_session_builder(role: str) -> bool:
    """True when an explicit builder is registered for this role (tests
    register scripted fakes; 'lead' is always present)."""
    return role in _SESSION_BUILDERS
```

Rationale: the host must know whether a verify episode CAN be served. In
production that means `model_provider is not None`; in scripted tests the
provider is None but a fake `episode:verify` builder is registered. The
availability predicate (§5.3) accepts either. Do NOT special-case test mode
anywhere else.

### 5.2 `src/investigation/host.py` — code-motion refactors (zero behavior change)

These three extractions are pure code motion; every existing test must pass
unmodified after this step alone.

**(a) `_snapshot_candidate_text`.** Split `_snapshot_content_hash`
(host.py:1134–1144) so the rendered text is reusable:

```python
def _snapshot_candidate_text(self, snapshot: dict[str, Any]) -> str:
    """Render one isolated episode snapshot with the canonical renderer."""
    from investigation.state import _restore_branch_into

    scratch = Workspace(
        cipher_text=self.workspace.cipher_text,
        plaintext_alphabet=self.workspace.plaintext_alphabet,
    )
    snap = copy.deepcopy(snapshot)
    _restore_branch_into(scratch, snap)
    return _decoded_text_for_panel(scratch, str(snap["name"]))

def _snapshot_content_hash(self, snapshot: dict[str, Any]) -> str:
    return _candidate_content_hash(self._snapshot_candidate_text(snapshot))
```

(`_branch_hash(ws, b)` is exactly
`_candidate_content_hash(_decoded_text_for_panel(ws, b))`, so this is
identical output.)

**(b) `_run_verify_episode`.** Extract from `_dispatch_verify_run` the block
host.py:666–695 (EpisodeSpec construction through the `episode_complete`
emit) into:

```python
def _run_verify_episode(
    self, *, candidate_text: str, goal: str, turn: int
) -> tuple[Any, float]:
    """Run one independent verify episode over ``candidate_text`` and do the
    standard bookkeeping (tool calls, episode budget, episode_complete emit).
    Returns (EpisodeResult, spend_usd). EpisodeSpec construction errors
    propagate (callers wrap)."""
```

Body: verbatim lines 667–670 (spec build, WITHOUT the try/except — see
below), 673–695 (provider resolve via
`self._provider_for_model((self._episode_models or {}).get("verify"))`,
`run_episode(...)` with the same kwargs, `episode_tool_calls` /
`episode_budget` extension, `spend_usd` computation, `episode_complete`
emit), then `return result, spend_usd`.

`_dispatch_verify_run` becomes:

```python
try:
    result, spend_usd = self._run_verify_episode(
        candidate_text=candidate, goal=str(args.get("goal") or ""), turn=turn
    )
except Exception as exc:  # noqa: BLE001 - bad spec -> structured error
    return json.dumps({"error": f"invalid verify episode: {exc}"})
```

(Behavior preserved: `run_episode` never raises ordinary exceptions — A9 —
and `KeyboardInterrupt` is a BaseException, so the widened try changes
nothing observable.)

**(c) `_attestation_record_from_result` + `_write_attestation`.** Split the
attestation block of `_dispatch_verify_run` (host.py:709–775):

```python
def _attestation_record_from_result(
    self, *, branch: str, content_hash: str, episode_result: dict[str, Any],
    episode_id: str, turn: int,
) -> AttestationRecord:
    """Pure field mapping from a verify episode result (same clamps as the
    dispatcher always applied: _clamp_coherence, clamp_unit_interval,
    normalize_damage_scope, normalize_repairability, strict is-True on
    reader_accepts_as_solution). Writes nothing."""
    # body = the AttestationRecord(...) constructor call at lines 710-736,
    # parameterized on branch/content_hash/episode_id/turn.

def _write_attestation(
    self, record: AttestationRecord, *, turn: int, seed_agenda: bool = True
) -> dict[str, Any]:
    """Append the record to state.verify_attestations and (optionally) run
    the Slice-6 agenda-seeding rule (non-positive + repairability ==
    'local_repair' mints open repair-agenda items bound to the record's
    branch/content_hash). Returns record.to_dict()."""
    # body = lines 737-775, using record.branch / record.content_hash /
    # record.anomalies etc.; the agenda block runs only when seed_agenda.
```

`_dispatch_verify_run` then does (replacing lines 709–775):

```python
if result.status == "ok" and isinstance(result.result, dict):
    record = self._attestation_record_from_result(
        branch=branch, content_hash=content_hash,
        episode_result=result.result, episode_id=result.episode_id, turn=turn,
    )
    self._write_attestation(record, turn=turn, seed_agenda=True)
    payload["attestation"] = { ... unchanged summary built from record ... }
```

The agenda block's `episode_id` field (line 772) must come from
`record.episode_id`; the `branch`/`content_hash`/`damage_scope`/
`repairability` fields from the record. Output byte-identical.

### 5.3 `src/investigation/host.py` — the arbitration helper (new code)

```python
def _verify_available(self) -> bool:
    """A verify episode can actually be served: a provider exists, or a
    scripted 'episode:verify' builder is registered (tests)."""
    from investigation.sessions import has_session_builder
    return self._model_provider is not None or has_session_builder("episode:verify")


def _arbitrate_repair(
    self, *, trigger_check: str, winner_snapshot: dict[str, Any],
    expected_hash: str, source_hash: str, turn: int, transaction_id: str,
) -> tuple[dict[str, Any], AttestationRecord | None]:
    """Run verifier arbitration for one mechanically-rejected repair.

    Returns (outcome, repaired_record). ``outcome`` is the acceptance
    sub-record's ``arbitration`` value; ``repaired_record`` is the
    AttestationRecord for the repaired content (None unless the verify ran
    and parsed). NEVER writes state — the caller writes the attestation
    only after a successful install (invariant: attestations name real,
    installed content)."""
    base: dict[str, Any] = {
        "requested": True, "engaged": True,
        "policy_id": ARBITRATION_POLICY_ID,
        "trigger_check": trigger_check,
        "constants": {
            "margin": ARBITRATION_MARGIN,
            "floor_target_language_confidence": ARBITRATION_FLOOR_TLC,
            "floor_semantic_recoverability": ARBITRATION_FLOOR_RECOVERABILITY,
        },
    }
    if not self._verify_available():
        outcome = {**base, "status": "unavailable",
                   "reason": "no_verification_provider"}
        self._emit("repair_arbitration", {
            "transaction_id": transaction_id, **outcome,
        }, outer_iteration=turn)
        return outcome, None
    if self.cost_ceiling_reached():
        outcome = {**base, "status": "unavailable",
                   "reason": "cost_ceiling_reached"}
        self._emit("repair_arbitration", {
            "transaction_id": transaction_id, **outcome,
        }, outer_iteration=turn)
        return outcome, None
    candidate_text = self._snapshot_candidate_text(winner_snapshot)
    repaired_hash = _candidate_content_hash(candidate_text)
    if repaired_hash != expected_hash:
        # Same renderer produced both digests; a mismatch means corruption.
        outcome = {**base, "status": "error",
                   "reason": "arbitration_render_mismatch",
                   "repaired_content_hash": repaired_hash,
                   "expected_content_hash": expected_hash}
        self._emit("repair_arbitration", {
            "transaction_id": transaction_id, **outcome,
        }, outer_iteration=turn)
        return outcome, None
    try:
        result, spend_usd = self._run_verify_episode(
            candidate_text=candidate_text, goal="", turn=turn
        )
    except Exception as exc:  # noqa: BLE001 - structured, never crashes the tx
        outcome = {**base, "status": "error",
                   "reason": f"invalid verify episode: {exc}"}
        self._emit("repair_arbitration", {
            "transaction_id": transaction_id, **outcome,
        }, outer_iteration=turn)
        return outcome, None
    base["episode_id"] = result.episode_id
    base["spend_usd"] = spend_usd
    base["repaired_content_hash"] = repaired_hash
    if result.status != "ok" or not isinstance(result.result, dict):
        outcome = {**base, "status": "error",
                   "reason": str(result.failure_reason or "episode_failed")}
        self._emit("repair_arbitration", {
            "transaction_id": transaction_id, **outcome,
        }, outer_iteration=turn)
        return outcome, None
    record = self._attestation_record_from_result(
        branch="", content_hash=repaired_hash,
        episode_result=result.result, episode_id=result.episode_id, turn=turn,
    )
    incumbent = latest_attestation_for_hash(
        self.state.verify_attestations, source_hash
    )
    accepted, rule, detail = _arbitration_verdict(record.to_dict(), incumbent)
    outcome = {
        **base,
        "status": "accepted" if accepted else "rejected",
        "rule": rule,
        "detail": detail,
        "repaired_attestation": record.to_dict(),
        "incumbent_attestation": dict(incumbent) if incumbent else None,
    }
    self._emit("repair_arbitration", {
        "transaction_id": transaction_id, "status": outcome["status"],
        "rule": rule, "trigger_check": trigger_check,
        "repaired_content_hash": repaired_hash,
        "incumbent_present": incumbent is not None,
        "episode_id": result.episode_id, "spend_usd": spend_usd,
    }, outer_iteration=turn)
    return outcome, record
```

Notes:

- `record.branch` is `""` at this point; the caller sets it to the installed
  name before writing (§5.4). The `repaired_attestation` dict embedded in the
  outcome therefore shows `branch: ""` on the REJECT path — that is correct:
  rejected-arbitration content never becomes a branch, and the attestation is
  preserved as evidence inside the transaction record only, never in
  `state.verify_attestations` (so it can never interact with DECL-1 hash
  matching, and no agenda items are minted for branch-less content). On the
  ACCEPT path the caller re-renders the sub-record after install (see §5.4)
  so the recorded `repaired_attestation.branch` equals the installed name.
- The verify episode itself appends its own `state.episode_ledger` entry and
  `state.budget_ledger` entries inside `run_episode` (existing behavior);
  `sync_budget()` remains the single source of truth. Nothing extra to do.

### 5.4 `src/investigation/host.py` — `validate_and_install_repair` changes

**Signature** (host.py:1353): add a keyword-only parameter with default:

```python
def validate_and_install_repair(
    self, *, tu, turn, branch, source_hash, att_key, pair, base_record,
    episode_payload, as_name, verifier_arbitration: bool = False,
) -> str:
```

**Local state** (next to `score_deltas` at line 1396):

```python
arbitration: dict[str, Any] | None = None
arbitration_record: AttestationRecord | None = None
```

**`_acceptance()`** (lines 1399–1409) becomes:

```python
def _acceptance() -> dict[str, Any]:
    payload = {
        "policy": (
            "default_deny_v1+verifier_arbitration_v1"
            if arbitration is not None and arbitration.get("engaged")
            else "default_deny_v1"
        ),
        "checks": acceptance_checks,
        "supported_forks": sorted(evidence["supported_forks"]),
        "edit_evidence_count": len(evidence["edit_evidence"]),
        "adjudicated": adjudicated_flag,
        "scores_before": before,
        "scores_after": after,
        "score_deltas": score_deltas,
    }
    if verifier_arbitration:
        payload["arbitration"] = (
            arbitration if arbitration is not None
            else {"requested": True, "engaged": False}
        )
    return payload
```

With `verifier_arbitration=False` (the default) the returned dict is
byte-identical to today (no `arbitration` key, `policy` unchanged) — this is
invariant 5 and is tested.

**Check 6** (lines 1515–1526) becomes:

```python
# Check 6 — collateral_within_limits (appended only when it can compare).
adj_summary = _winner_adjudication_summary(episode_calls, winner) or {}
damaged = adj_summary.get("damaged_occurrences")
improved = adj_summary.get("improved_occurrences")
collateral_entry: dict[str, Any] | None = None
collateral_failed = False
if isinstance(damaged, (int, float)) and isinstance(improved, (int, float)):
    collateral_ok = damaged <= improved
    collateral_entry = {
        "check": "collateral_within_limits", "passed": collateral_ok,
        "damaged_occurrences": damaged, "improved_occurrences": improved,
    }
    acceptance_checks.append(collateral_entry)
    if not collateral_ok:
        if not verifier_arbitration:
            return _fail("materially_non_improving")
        # Arbitration requested: fall through so the probe runs, checks 7/8
        # are appended, scores_after/score_deltas are recorded (fixing the
        # null-deltas observability gap the motivating artifact shows), and
        # the arbitration verdict decides the outcome after check 8.
        collateral_failed = True
```

**Checks 7 and 8** (lines 1528–1570): the probe block (1529–1553) is
unchanged — it now also runs on the collateral-failed arbitration path
(winner snapshot lookup, `_probe_snapshot_scores`, delta computation,
`no_op_probe` append). The `no_op` failure still returns `_fail("no_op")`
unconditionally — **arbitration never rescues a no-op** (it is also
unreachable when check 6 failed, because `winner ∈ changed` implies the
snapshot hash differs from `source_hash`; keep the defensive check anyway).

Check 8 (1555–1570) becomes:

```python
# Check 8 — scalar_non_decrease (default deny on any measured decrease).
decreased = (
    (dict_delta is not None and dict_delta < 0)
    or (quad_delta is not None and quad_delta < 0)
)
# Default deny: any measured scalar decrease rejects. REPAIR_ACCEPTANCE_POLICY
# is the M5.4 hook; no allow-policy branch is implemented yet, so the guard
# asserts the invariant rather than carrying a dead alternative. Verifier
# arbitration below is NOT that policy object: it is an independent-reader
# evidence source (docs/specs/verifier_arbitrated_repair_spec.md), and it
# only ever OVERRULES a reject, never replaces the default.
assert REPAIR_ACCEPTANCE_POLICY is None
scalar_ok = not decreased
scalar_entry = {
    "check": "scalar_non_decrease", "passed": scalar_ok,
    "deltas": dict(score_deltas),
}
acceptance_checks.append(scalar_entry)
if collateral_failed or not scalar_ok:
    if not verifier_arbitration:
        return _fail("materially_non_improving")
    trigger = (
        "collateral_within_limits" if collateral_failed
        else "scalar_non_decrease"
    )
    arbitration, arbitration_record = self._arbitrate_repair(
        trigger_check=trigger, winner_snapshot=winner_snapshot,
        expected_hash=changed[winner], source_hash=source_hash,
        turn=turn, transaction_id=base_record["transaction_id"],
    )
    if arbitration.get("status") != "accepted":
        return _fail("materially_non_improving")
    if collateral_entry is not None and not collateral_entry["passed"]:
        collateral_entry["overruled_by_arbitration"] = True
    if not scalar_ok:
        scalar_entry["overruled_by_arbitration"] = True
```

(Note `winner_snapshot` is already resolved at line 1529 before the probe;
the arbitration call sits after check 8's append, so the acceptance record
always carries all evaluated checks plus real `scores_after`/`score_deltas`
whenever arbitration engaged — including the check-6 trigger case where
today's record has nulls.)

**Install path** (lines 1572–1639): replace the single early
`acceptance = _acceptance()` snapshot at line 1572 with per-use-site calls
(the closure over mutable locals makes a stale snapshot fragile once
arbitration mutates state):

- In the `install_failed` branch (1584–1596): if arbitration was accepted,
  first set `arbitration["attestation_recorded"] = False`; then build the
  record with `"acceptance": _acceptance()`.
- After a successful install (after `result_hash` is computed at 1597),
  insert:

```python
if (
    arbitration is not None
    and arbitration.get("status") == "accepted"
    and arbitration_record is not None
):
    if result_hash == arbitration_record.content_hash:
        arbitration_record.branch = installed
        self._write_attestation(arbitration_record, turn=turn, seed_agenda=True)
        arbitration["attestation_recorded"] = True
        arbitration["repaired_attestation"] = arbitration_record.to_dict()
    else:
        # Defensive only: install restored the SAME snapshot the arbitration
        # rendered, and dedup only merges content-identical branches, so a
        # mismatch means corruption. Do not write a mislabeled attestation.
        arbitration["attestation_recorded"] = False
```

- Then build the installed `record` with `"acceptance": _acceptance()`
  (all other fields unchanged, including `"reverification_required": True` —
  the flag stays for audit parity; workflow consumers read attestations by
  hash, and on the arbitration-accept path a fresh hash-matched attestation
  now exists, which is the point).

Consequences to document in a code comment: when arbitration accepted via
rule 1 (`reader_accepts_as_solution`), the installed branch immediately
satisfies DECL-1 (a genuine server-run attestation of exactly this content
exists) — this is intended and identical in trust terms to running
`request_independent_verification` right after the install. When accepted
via rules 2/3 with a non-positive verdict and `repairability ==
"local_repair"`, `_write_attestation` seeds repair-agenda items for the
INSTALLED branch — also intended (the reader's residual anomalies are real).

### 5.5 `src/investigation/host.py` — `_dispatch_repair_transaction` (v3 path)

At the `validate_and_install_repair` call (lines 1693–1698), add:

```python
verifier_arbitration=bool(args.get("verifier_arbitration")),
```

Everything else (preconditions, worker episode, failure handling) is
untouched. The v3 REP-7 phase gate in `handle_tool` is untouched.

### 5.6 `src/investigation/episodes.py` — v3 lead tool schema

`REPAIR_TRANSACTION_TOOL` (line ~785): add to `properties` (NOT to
`required`):

```python
"verifier_arbitration": {
    "type": "boolean",
    "description": (
        "Opt-in (default false): if only the collateral/scalar scoring "
        "checks would reject, run one fresh server-side independent verify "
        "on the repaired fork; it installs only if the independent reader "
        "judges it strictly better than the incumbent (or accepts it as a "
        "solution). Evidence-binding checks are never arbitrable. Paid; "
        "unavailable without a verify provider (typed fallback to the "
        "mechanical reject)."
    ),
},
```

### 5.7 `src/mcp_server/tools.py` — MCP tool schema

In the `repair_transaction` definition (5.18, lines 350–387):

1. Add the SAME `verifier_arbitration` property (identical text as §5.6) to
   `properties`, not `required`.
2. Append one sentence to the tool `description`: `" Optional "
   "verifier_arbitration=true: when only the collateral/scalar scoring "
   "checks would reject, one fresh server-side independent reader "
   "arbitrates the repaired fork and it installs only if judged strictly "
   "better; keyless servers return a typed unavailable fallback to the "
   "mechanical reject."`

### 5.8 `src/mcp_server/repair.py` — pass-through

In `dispatch_repair_transaction` (line 185): the `tu` construction (lines
191–194) already forwards all args except
`investigation_id`/`expected_revision`, so `verifier_arbitration` lands in
the recorded ToolCall arguments automatically — no change there. At the
`host.validate_and_install_repair` call (lines 272–277), add:

```python
verifier_arbitration=bool(args.get("verifier_arbitration")),
```

No other MCP server change: `server.py` routing, revision/commit protocol,
`verify.py` are untouched. (The keyless typed result comes from the host's
`_verify_available()`, which is False exactly when the runtime was built
with `verify_provider=None` and no scripted builder exists.)

## 6. Typed outcomes and the acceptance sub-record

`acceptance.arbitration` (present iff `verifier_arbitration=true` was
passed):

| Shape | Meaning | Transaction outcome |
|---|---|---|
| `{"requested": true, "engaged": false}` | Flag passed, but no scoring reject occurred (installed via checks), or the transaction failed on a non-arbitrable check (1–5, 7) | unchanged from today |
| `status: "unavailable", reason: "no_verification_provider"` | keyless | `failed` / `materially_non_improving` (as today) |
| `status: "unavailable", reason: "cost_ceiling_reached"` | BUD-1 | `failed` / `materially_non_improving` |
| `status: "error", reason: ...` | render mismatch, spec error, or episode failure | `failed` / `materially_non_improving` |
| `status: "rejected", rule: null, detail: {...}` | reader did not prefer the repaired fork | `failed` / `materially_non_improving`; counts toward saturation exactly as today |
| `status: "accepted", rule: "reader_accepts_as_solution" \| "margin_improvement" \| "absolute_floor"` | reader prefers the repaired fork | `installed`; failed scoring check entries carry `overruled_by_arbitration: true`; `attestation_recorded: true` and the attestation (branch = installed name, hash = repaired content) is in `state.verify_attestations` |

Engaged outcomes always also carry: `policy_id`, `trigger_check`,
`constants` (the three pre-registered numbers), and — once the verify ran —
`episode_id`, `spend_usd`, `repaired_content_hash`,
`repaired_attestation`, `incumbent_attestation` (dict or null).

The acceptance `policy` field is `"default_deny_v1+verifier_arbitration_v1"`
iff arbitration ENGAGED (any of the last four rows), else
`"default_deny_v1"`.

Events: one `repair_arbitration` event per engagement (payload per §5.3);
the arbitration verify additionally emits the standard `episode_complete`
(kind `verify`) and appears in `state.episode_ledger` like any verify
episode.

## 7. Failure-classification and saturation semantics (unchanged, verify in tests)

- Arbitration-rejected/unavailable/error transactions fail with reason
  `materially_non_improving` → `failure_class: "evidence"`,
  `counted_evidence_failure: true`, saturation `evidence_failures` +1,
  pair added to `evidence_failed_pairs` (`_settle_repair_outcome`,
  host.py:1147–1192). No new reason strings are added to
  `_EVIDENCE_FAILURE_REASONS`.
- `check_repair_preconditions` is untouched. REP-2 duplicate suppression and
  `pair_evidence_failed` therefore bound arbitration spend to one verify per
  (content, interpretation) pair.

## 8. Case-2 scope notes (light touches only — NOT part of the core change)

1. **Batch/scattered confirmation test** (§9, test T11): the acceptance
   pipeline and arbitration operate on whole-fork CONTENT
   (`source_hash` → winner snapshot hash); nothing is span-local. A single
   `hypothesis_test_words` hypothesis whose key edits propagate to several
   scattered occurrences (one damaged symbol appearing in multiple words) is
   one changed finalist fork, and the reader arbitrates the whole repaired
   fork. The test makes this explicit so "distributed damage" is provably
   installable through the existing surface.
2. **Doctrine line** (exact text, added in three places per §10.3/§10.4):
   *"Distributed damage that is a set of individually-simple key errors is
   still batch-repairable via `repair_hypotheses_test` →
   `repair_transaction`; do not treat `distributed` automatically as
   broaden-only."* The damage-scope→route mapping is ledger row WF-4 and is
   ADVISORY in the MCP surface, so this is guidance, not a gate change. No
   change to `context.py` routing, thresholds, or `workflow_state`.

## 9. Required tests

New file `tests/test_repair_arbitration.py` (v3/host level) plus additions
to `tests/test_mcp_tools.py` (MCP level). Reuse
`tests/support/scripted_v3.py` (`keyed_catton_state`, `seed_reading`,
`seed_negative_attestation`, `register_programmable_repair`,
`make_verify_builder`, `ScriptedSession`) and `tests/support/mcp.py`
(`make_server`, `call`, `start`). The v3 driver pattern is
`test_loop_v3.py::_run_single_repair` (lines ~1265–1280): replicate a local
variant that accepts the tool `input` dict (so
`verifier_arbitration: True` can be passed) and does NOT auto-register the
positive `verify_fake` (register the arbitration verdict explicitly per
test via `sessions_mod.register_session_builder("episode:verify",
make_verify_builder(VERDICT))`, popped in `finally`).

The scalar-decrease trigger program is the existing one from
`test_s4_scalar_decrease_default_denied`: reading/apply text `"COTON"`
against `keyed_catton_state` (decode `CATON`), which deterministically
yields `quad_delta < 0`.

Baseline invariance (invariant 5): **no existing test file is modified.**
`tests/test_loop_v3.py`, `tests/test_mcp_tools.py` (existing tests),
`tests/test_investigation_state.py`, `tests/test_v3_sequence_b.py` and the
rest of the suite pass unmodified: full suite = 1753 + (new tests) passed /
2 skipped.

**T1 — mechanical-reject + arbitration-accepts → installs (margin rule).**
v3 level. `seed_negative_attestation` on main (incumbent tlc 0.8 / rec 0.7),
COTON program, verdict `{tlc: 0.9, rec: 0.8, reader_accepts_as_solution:
False, damage_scope: "local", repairability: "local_repair", anomalies:
["x"], coherence: 6, reader_accepts: False, gloss: "...", uncertainty_note:
""}`, input `{"branch": "main", "as_name": "transaction_repaired",
"verifier_arbitration": True}`. Assert: transaction `installed`; branch
`transaction_repaired` exists; `acceptance.policy ==
"default_deny_v1+verifier_arbitration_v1"`; `acceptance.arbitration.status
== "accepted"`, `rule == "margin_improvement"`, `trigger_check ==
"scalar_non_decrease"`, `attestation_recorded is True`; the
`scalar_non_decrease` check entry has `passed is False` and
`overruled_by_arbitration is True`; `state.verify_attestations[-1]` has
`branch == "transaction_repaired"` and `content_hash ==
_branch_hash(workspace, "transaction_repaired")`; an open agenda item for
the installed branch exists (non-positive + local_repair seeding); the
episode ledger contains kinds `["repair", "verify"]`.

**T2 — mechanical-reject + arbitration-rejects → fails.** Same as T1 but
verdict `{tlc: 0.75, rec: 0.65, ...}` (below incumbent). Assert: status
`failed`, reason `materially_non_improving`, `failure_class == "evidence"`,
`counted_evidence_failure is True`, saturation `evidence_failures == 1`;
`acceptance.arbitration.status == "rejected"`, `rule is None`, and
`repaired_attestation`/`incumbent_attestation` both present; branch NOT
installed; `len(state.verify_attestations) == 1` (only the seeded incumbent
— the arbitration attestation is NOT written to state); no agenda items for
the repaired content hash.

**T3 — flag default false → today's behavior byte-identical.** Run the T1
fixture WITHOUT `verifier_arbitration` (and without registering a verify
builder). Assert the transaction fails exactly as
`test_s4_scalar_decrease_default_denied` records it AND
`set(tx["acceptance"].keys()) == {"policy", "checks", "supported_forks",
"edit_evidence_count", "adjudicated", "scores_before", "scores_after",
"score_deltas"}` (no `arbitration` key) and `acceptance["policy"] ==
"default_deny_v1"`; no verify episode ran. (The unmodified existing repair
tests are the broader half of this guarantee.)

**T4 — keyless → typed unavailable, mechanical reject preserved.** T1
fixture with flag True but NO registered `episode:verify` builder and no
provider (scripted `run_v3` has `model_provider=None`). Assert: failed /
`materially_non_improving`; `acceptance.arbitration == {"requested": True,
"engaged": True, "policy_id": ..., "trigger_check": ..., "constants": ...,
"status": "unavailable", "reason": "no_verification_provider"}`; no episode
of kind `verify` in the ledger; spend unchanged.

**T5 — genuinely-worse repair still fails arbitration (regression).** No
incumbent attestation; COTON program (a mechanically-worse fork); verdict
`{tlc: 0.30, rec: 0.20, reader_accepts_as_solution: False, ...}`. Assert
rejected via the floor rule (`status == "rejected"`), transaction failed,
nothing installed, nothing attested. (This pins the MECHANISM: a reader
that does not prefer the fork cannot install it — invariant 4.)

**T6 — floor rule replica of the motivating case.** No incumbent; verdict
`{tlc: 0.97, rec: 0.88, reader_accepts_as_solution: False, damage_scope:
"local", repairability: "local_repair", anomalies: [], ...}`. Assert
installed with `rule == "absolute_floor"`.

**T7 — `_arbitration_verdict` unit tests** (pure function): (a)
`reader_accepts_as_solution` short-circuits regardless of scalars; (b)
margin boundary: deltas (+0.03, +0.02) accepted (sum == 0.05), (+0.03,
+0.019) rejected; (c) non-monotone rejected: (+0.20, −0.01); (d) legacy
incumbent without the `target_language_confidence` key routes to the floor
rule; (e) floor boundaries: (0.90, 0.60) accepted, (0.899, 0.60) rejected,
(0.90, 0.599) rejected; (f) missing scalar fields in the repaired dict
clamp to 0.0 and reject.

**T8 — arbitration never engages on evidence checks.** Fabricated-winner
program (`test_s4_fabricated_winner_rejected` shape) with flag True and a
registered ACCEPTING verify builder. Assert: failed `unsupported_winner`;
no verify episode ran; `acceptance.arbitration == {"requested": True,
"engaged": False}`.

**T9 — check-6 trigger records full deltas and installs.** Host-level
white-box (no MCP, no run_v3): build an `InvestigationHost` over
`keyed_catton_state`-style state; synthesize what `mcp_server/repair.py`
synthesizes — an episode-ledger entry (`kind: "repair_compile"`) whose
`branch_snapshots` contain one changed winner snapshot, plus one
`host.episode_tool_calls` ToolCall with `tool_name="hypothesis_test_words"`
and a JSON result `{"status": "ok", "items": [{"status": "ok",
"installed_fork": WINNER, "edits": [...], "adjudication_summary":
{"damaged_occurrences": 3, "improved_occurrences": 0}}], "finalists":
[{"installed_fork": WINNER, "edits": [...]}]}` — then call
`host.validate_and_install_repair(..., episode_payload={"episode_id": ...,
"status": "ok", "result": {"applied": True, "best_branch": WINNER, "edits":
[...], "verdicts": [], "collateral": {}}}, verifier_arbitration=True)` with
an accepting verify builder registered. Assert: `trigger_check ==
"collateral_within_limits"`; the acceptance record contains ALL of
`collateral_within_limits` (failed, overruled), `no_op_probe` (passed),
`scalar_non_decrease` entries AND non-null
`scores_after`/`score_deltas` (the motivating artifact's null-deltas gap is
closed on the arbitrated path); installed. Also run the same fixture
flag-false and assert the record stops at check 6 with `scores_after is
None` (today's shape preserved).

**T10 — MCP end-to-end (case-1 replica).** In `tests/test_mcp_tools.py`.
Fixture: plaintext `"THE MISSING TRAWLER RESTED IN THE COVE"`-style basin
(lowercase-symbol cipher as `apply_basin` does), keyed correct EXCEPT the
`w` symbol mapped to `I` so word 2 decodes `TRAILER` (in the common list;
the true word `TRAWLER` is not; the `w` symbol must occur only in that
word). Flow: `reading_record` → `repair_hypotheses_test` with `{"word":
"TRAWLER", "word_index": 2}` → `repair_transaction` with
`verifier_arbitration=True` on a server built with `verify="dummy"` and
`make_verify_builder({tlc: 0.97, rec: 0.88, reader_accepts_as_solution:
False, damage_scope: "local", repairability: "local_repair", anomalies: [],
...})`. The test MUST first assert the mechanical trigger actually fired
(the acceptance checks contain a failed `collateral_within_limits` or
`scalar_non_decrease` entry) so the fixture is self-validating, then assert
`status == "installed"`, `acceptance.arbitration.status == "accepted"`, and
the installed branch's `candidate_show` attestation history contains the
arbitration attestation. Variants in the same file: (a) same fixture,
keyless server (`verify="none"`), flag True → failed with
`acceptance.arbitration.status == "unavailable"`, reason
`no_verification_provider`; (b) same fixture, flag absent → failed
`materially_non_improving` with NO `arbitration` key (today's shape); (c)
same fixture, rejecting verdict `{tlc: 0.4, rec: 0.3}` → failed,
`arbitration.status == "rejected"`.

**T11 — batch repair of SCATTERED errors arbitrates the whole fork (case-2
light touch).** MCP or host level: a basin where ONE damaged symbol occurs
in ≥3 words scattered across the text, each decoding to an in-list wrong
word while the corrected forms are off-list (TRAWLER-style); one
`hypothesis_test_words` hypothesis on one of the words produces a single
changed finalist fork whose key edit fixes all occurrences. Assert the
mechanical counter reports `damaged_occurrences > improved_occurrences`
(self-validating trigger), arbitration (accepting reader) installs the
fork, and the installed decode shows ALL scattered corrections. The
implementer picks the concrete words; the in-test trigger assertions make
the fixture fail loudly if the choice does not trigger.

**T12 — code-motion invariance.** After §5.2 alone (before wiring the
flag), the full suite passes unmodified — implement and run in that order.
No dedicated new test; this is a sequencing requirement.

## 10. Documentation and ledger edits

### 10.1 `docs/mcp_policy_provenance_ledger.md`

This change is a **pre-registered contract change recorded in the ledger**
(the CMP-2 pattern: named, evidenced, never silent) — it does NOT remove
default-deny, so the §6 ablation bar is not being claimed; the two-case
artifact is the pre-registered evidence that moves the SCORING-POLICY half
of REP-3/REP-4 from "hard-only" to "hard OR verifier-arbitrated".

1. Key-artifacts paragraph (lines ~44–49): append
   `docs/evidence/c56de7e6c600_repair_guard_false_reject.json` (repair-guard
   false reject + distributed-routing miss, 2026-07-17) to the artifact
   list.
2. **REP-3 row**, "MCP v1 form" cell — replace
   `**Hard** — reused as-is for MCP's `repair_transaction` over
   client-compiled finalists (§3.5)` with:
   `**Hard** — reused as-is for MCP's `repair_transaction` (§3.5). The two
   SCORING checks (collateral_within_limits, scalar_non_decrease) may be
   overruled ONLY by opt-in verifier arbitration — a server-run, hash-bound
   independent verify on the repaired fork (`verifier_arbitration=true`,
   default false; spec `docs/specs/verifier_arbitrated_repair_spec.md`;
   evidence `docs/evidence/c56de7e6c600_repair_guard_false_reject.json`).
   Checks 1–5 and the no-op probe are never arbitrable.`
3. **REP-4 row**, "MCP v1 form" cell — replace
   `**Hard**, unchanged for clean comparison; record every scalar-denied
   install as telemetry so the assumption is auditable` with:
   `**Hard default**, opt-in verifier-arbitrated: the POL half ("any scalar
   decrease is materially bad") acquired its pre-registered counter-evidence
   (the c56de7e6c600 false reject — objectively correct corrections denied
   on a dictionary-membership count while an independent reader scored the
   corrected reading 0.97) and may now be overruled per-call by a server-run
   independent verify (`verifier_arbitration`, default false; arbitration
   decision + both attestations recorded in the acceptance sub-record). The
   INV half (default-deny; nothing installs on the model's say-so) is
   unchanged; `REPAIR_ACCEPTANCE_POLICY` remains the untouched M5.4 hook.`
4. Section-3 "Change-evidence" paragraph (lines ~109–113): append one
   sentence: `The REP-3/REP-4 scoring-policy half met its evidence bar on
   2026-07-17 (artifact
   `docs/evidence/c56de7e6c600_repair_guard_false_reject.json`, two cases);
   the recorded change is the opt-in verifier-arbitration seam
   (`docs/specs/verifier_arbitrated_repair_spec.md`), not a removal of
   default-deny.`
5. Do NOT change any row's classification or the §Summary counts —
   classifications are unchanged (REP-4 stays INV/POL straddle).

### 10.2 `TOOLS.md`

MCP table row (line 1701): change the `repair_transaction` purpose cell to
`Host-validated install of one compiled winner (opt-in
`verifier_arbitration` lets an independent reader overrule a
collateral/scalar reject).` No other TOOLS.md change (the v3 lead
`repair_transaction` tool has no per-tool section there today; do not add
one).

### 10.3 `docs/mcp_onboarding.md`

1. §2 doctrine bullet "Read, then repair, then reverify." (lines ~72–77):
   after "The host rejects unsupported edits and any scoring regression;"
   insert: "if you are confident the mechanical counter is wrong (correct
   words outside the common list), pass `verifier_arbitration=true` — an
   independent reader then arbitrates the repaired fork, and only a reading
   it judges strictly better installs;". Keep the rest of the bullet.
2. Same §2 section: add the case-2 doctrine line from §8.2 verbatim as its
   own sentence at the end of that bullet (with the WF-4-advisory note:
   "(damage-scope routing is advisory — WF-4)").
3. §3 tool list line for `repair_transaction` (line ~115): append
   "(supports opt-in `verifier_arbitration`)".

### 10.4 `AGENTS.md`

In the "Cracking a cipher (MCP quick path)" section (after the
`docs/mcp_onboarding.md` pointer paragraph), add the same doctrine line from
§8.2 verbatim, prefixed "Repair doctrine:". One line only.

## 11. Non-goals / out of scope

- No change to WF-4 routing, thresholds, `workflow_state`, or
  `allowed_episode_kinds` (case 2 is doctrine + a test, per §8).
- No change to `check_repair_preconditions`, saturation accounting, failure
  taxonomy, or `_EVIDENCE_FAILURE_REASONS`.
- No `REPAIR_ACCEPTANCE_POLICY` implementation; the `None` assert stays.
- No v2-loop changes; no benchmark-runner changes; no changes to
  `verify.py`/`server.py` beyond what §5.7–5.8 lists (i.e., none).
- No auto-arbitration: the host never arbitrates unless the caller passed
  the flag on THAT call.
- No incumbent-minting: arbitration never runs a verify on the PRE-repair
  content (that would mint a new attestation key and reset saturation — the
  documented `attestation_key` laundering seam). The incumbent is whatever
  attestation already exists for `source_hash`, else the floor rule.

## 12. Residual risk (accepted, documented)

Under the floor rule (no incumbent), a repair the mechanical counter
correctly dislikes could install if an independent reader scores the
repaired text ≥0.90/≥0.60 — but that region is precisely the false-reject
class this spec exists to fix (correct rare words off the 5,000-word list),
the floors sit well above the WF-4 "high" thresholds, engagement requires an
explicit per-call opt-in, every engagement is fully recorded (both
attestations, rule, constants, deltas), and the installed fork still carries
`reverification_required` plus normal DECL-1 gating. T5 pins the rejection
mechanism.

## 13. Implementation order and acceptance

1. §5.1 + §5.2 (code motion) → full suite green, unmodified (T12).
2. §4 + §5.3 + §5.4 + §5.5 + §5.6 (host + v3 flag) → T1–T9 green.
3. §5.7 + §5.8 (MCP surface) → T10–T11 green.
4. §10 docs/ledger edits.
5. Full suite: 1753 existing tests still pass unmodified (2 skipped
   unchanged); new tests all green. Anything requiring an edit to an
   existing test is a spec violation — stop and report instead.
