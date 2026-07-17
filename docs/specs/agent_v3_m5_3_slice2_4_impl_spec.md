# M5.3 Slices 2 + B2 + 4 — Implementation Sub-Spec (repair saturation, identity rename, host-validated acceptance)

Status: final, ready to implement. Authored 2026-07-17 against HEAD `24541d2`
(Slices 1, 3, 5 already landed).

Authority: `docs/specs/agent_v3_m5_3_control_reliability_spec.md` — Slice 2
(lines 276–338), Slice 4 (lines 387–416) — and Amendment B2 in
`docs/repair_reframe_m53_comments.md` (lines 279–286). This document refines
those sections into exact code edits; where the master spec leaves a choice,
this document makes it. Implement THIS document. Do not invent scope beyond
it; a genuine gap goes back to the spec author.

Line numbers below are HEAD-`24541d2` line numbers, always accompanied by an
anchor snippet. Match on the snippet, not the number.

Baseline: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` currently
reports **1608 passed, 1 failed** — the red test is
`tests/test_lead_context.py::test_negative_partial_attestation_creates_repair_action_menu`,
which this slice turns green (§7.1). After implementation the full suite must
be green including every test in §12.

Files touched (production):

| File | What changes |
|---|---|
| `src/investigation/state.py` | new `repair_saturation` field + serialization; identity helpers (`attestation_key`, `saturation_key`, `pair_digest`, `latest_attestation_for_hash`, `new_saturation_entry`, `get_or_create_saturation_entry`) |
| `src/investigation/reading.py` | new `interpretation_digest(reading_dict)` helper (B2) |
| `src/investigation/context.py` | `repair_exhausted` workflow state + menu; fail-closed unknown phase; repair_required wording fix; `_fresh_attestation` delegates to shared helper |
| `src/investigation/loop_v3.py` | saturation gates + counters, reading suppression, reason taxonomy/classifier, Slice-4 acceptance validator, record-shape extensions, pending-experiment pointer, live episode-kind gate |

No changes to `episodes.py`, `actions.py`, `experiments.py`, TOOLS.md (no v2
tool changed), or the artifact schema (`ToolCall` already carries
`episode_id`).

---

## 1. Shared identity primitives

### 1.1 `interpretation_digest` (B2) — `src/investigation/reading.py`

Add a module-level function (place it directly after `new_reading_id`):

```python
def interpretation_digest(reading: dict[str, Any]) -> str:
    """Content digest of a stored Reading dict (M5.3 Slice 2 / B2).

    Two readings with byte-identical machine-actionable content on the same
    candidate are the SAME interpretation regardless of reading_id, source,
    created_turn, or wording of the goal that produced them (master spec
    Design Principle 3). Includes per-fragment confidence (it changes
    applicability via MIN_REPAIR_FRAGMENT_CONFIDENCE) and the bound candidate
    hash; excludes reading_id, source, created_turn, and overall_confidence
    (advisory only). In M5.3 the interpretation is always a legacy Reading;
    M5.4 InterpretationPackets reuse this seam without a state migration.
    """
    fragments = []
    for f in reading.get("fragments") or []:
        if not isinstance(f, dict):
            continue
        fragments.append({
            "text": f.get("text"),
            "repair_text": f.get("repair_text"),
            "span_id": f.get("span_id"),
            "token_indices": f.get("token_indices"),
            "start": f.get("start"),
            "end": f.get("end"),
            "confidence": f.get("confidence"),
        })
    payload = {
        "candidate_content_hash": reading.get("candidate_content_hash"),
        "fragments": fragments,
        "holes": [str(h) for h in reading.get("holes") or []],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
```

`reading.py` must import `hashlib` and `json` (json is already imported;
add hashlib if absent).

### 1.2 Key helpers — `src/investigation/state.py`

Add module-level functions (after the `AttestationRecord` class; they need
only `hashlib`/`json` — add those imports):

```python
def attestation_key(attestation: dict[str, Any] | None) -> str:
    """Identity of one verifier-evidence unit (M5.3 Slice 2).

    Primary component is the verify episode id (AttestationRecord has no id
    field; episode_id is unique per verify episode). Fallback for records
    without an episode_id (hand-seeded tests, legacy data): a digest over the
    verdict content (anomalies + coherence + reader_accepts). "none" when no
    attestation exists for the candidate.

    Known seam (documented, not built): keying by episode_id means a re-verify
    of unchanged content mints new verifier evidence and resets saturation.
    Re-verification of unchanged content is already discouraged by the system
    prompt and workflow hints; if paid runs show saturation laundering via
    re-verification, switch the primary component to the content digest below
    (a one-line change; both forms are specified here).
    """
    if not attestation:
        return "none"
    episode_id = str(attestation.get("episode_id") or "")
    if episode_id:
        return f"ep:{episode_id}"
    payload = {
        "anomalies": [str(a) for a in attestation.get("anomalies") or []],
        "coherence": int(attestation.get("coherence") or 0),
        "reader_accepts": bool(attestation.get("reader_accepts")),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"digest:{digest}"


def saturation_key(candidate_content_hash: str, att_key: str) -> str:
    """Key of one repair-saturation entry: (candidate content, verifier evidence).

    A new candidate hash OR genuinely new verifier evidence yields a new key —
    saturation reset is automatic (a fresh entry). A reworded goal changes
    neither component, so it never resets saturation.
    """
    return hashlib.sha1(
        f"{candidate_content_hash}|{att_key}".encode("utf-8")
    ).hexdigest()


def pair_digest(source_content_hash: str, interp_digest: str) -> str:
    """Identity of one evaluated source/interpretation pair.

    DIGEST-based (not reading_id-based): re-running a byte-identical reading
    under a fresh reading_id or a new as_name is the same pair.
    """
    return hashlib.sha1(
        f"{source_content_hash}|{interp_digest}".encode("utf-8")
    ).hexdigest()


def latest_attestation_for_hash(
    attestations: list[dict[str, Any]], content_hash: str
) -> dict[str, Any] | None:
    """Newest attestation matching ``content_hash`` by (created_turn, episode_id).

    Single shared selection rule — context._fresh_attestation and the
    repair-transaction dispatcher must both use THIS function (the dispatcher
    previously tie-broke on created_turn only)."""
    matches = [a for a in attestations if a.get("content_hash") == content_hash]
    return max(
        matches,
        key=lambda a: (int(a.get("created_turn") or 0), str(a.get("episode_id") or "")),
        default=None,
    )


def new_saturation_entry(
    candidate_content_hash: str, att_key: str, turn: int
) -> dict[str, Any]:
    """Fresh repair-saturation entry (all JSON-native; see field table)."""
    return {
        "candidate_content_hash": candidate_content_hash,
        "attestation_key": att_key,
        "evidence_failures": 0,
        "process_failures": {},
        "evidence_failed_pairs": [],
        "finalist_hashes": [],
        "readings": 0,
        "exhausted": False,
        "pending_experiment_id": None,
        "created_turn": turn,
        "updated_turn": turn,
    }


def get_or_create_saturation_entry(
    state: "InvestigationState",
    candidate_content_hash: str,
    att_key: str,
    turn: int,
) -> dict[str, Any]:
    key = saturation_key(candidate_content_hash, att_key)
    entry = state.repair_saturation.get(key)
    if entry is None:
        entry = new_saturation_entry(candidate_content_hash, att_key, turn)
        state.repair_saturation[key] = entry
    return entry
```

Mutation rule: only `loop_v3.py` calls `get_or_create_saturation_entry` /
mutates entries. `context.py` READS entries via
`state.repair_saturation.get(...)` and never creates or mutates one
(`build_lead_context` stays a pure function).

### 1.3 `_fresh_attestation` delegates — `src/investigation/context.py`

Replace the body of `_fresh_attestation` (line 625, anchor
`matches = [`) with:

```python
def _fresh_attestation(
    state: InvestigationState, branch: str
) -> dict[str, Any] | None:
    from investigation.state import latest_attestation_for_hash

    return latest_attestation_for_hash(
        state.verify_attestations, _branch_content_hash(state, branch)
    )
```

(Behavior is identical; this is de-duplication so the loop and the context
can never disagree about which attestation is "latest".)

---

## 2. Durable saturation state — `src/investigation/state.py`

### 2.1 Field

Add to `InvestigationState`, directly after `repair_transactions`
(line 244, anchor `repair_transactions: list[dict[str, Any]] = field(...)`):

```python
    # M5.3 Slice 2: durable repair-saturation counters, keyed by
    # saturation_key(candidate_content_hash, attestation_key). Keying by
    # (content, evidence) makes reset-on-new-content/new-evidence automatic
    # (a new pair is a fresh entry) and makes reworded goals irrelevant (the
    # goal is not in the key). Entries are created/mutated ONLY by loop_v3;
    # context.py reads them. Absent from pre-M5.3 artifacts -> empty on load.
    repair_saturation: dict[str, dict[str, Any]] = field(default_factory=dict)
```

### 2.2 Entry schema (normative)

| Field | Type | Meaning |
|---|---|---|
| `candidate_content_hash` | `str` | the candidate hash component of the key (denormalized for review) |
| `attestation_key` | `str` | the verifier-evidence component (`ep:<episode_id>`, `digest:<sha256>`, or `none`) |
| `evidence_failures` | `int` | count of transactions that consumed saturation budget (evidence-class failures, including reclassified second process failures). `>= 2` latches `exhausted` |
| `process_failures` | `dict[str, int]` | pair_digest → count of process-class failures for that pair. `1` = the single linked retry is available; a second process failure for the pair is counted as an evidence failure |
| `evidence_failed_pairs` | `list[str]` | sorted, deduped pair_digests that were evidence-evaluated and failed; such a pair can NEVER rerun (any as_name, any fresh reading_id with identical content) |
| `finalist_hashes` | `list[str]` | sorted, deduped content hashes of every changed finalist any transaction under this key generated (evidence for the "compare genuinely distinct finalists" menu action; no gating logic reads it in M5.3) |
| `readings` | `int` | reading episodes completed for this (content, evidence); cap 1 — repeat readings are suppressed (§5) |
| `exhausted` | `bool` | durable latch; set when `evidence_failures >= 2`; never cleared (escape = new content or new evidence = new key) |
| `pending_experiment_id` | `str \| None` | experiment submitted while this entry was exhausted (§8); validated against the queue at render time, never cleared (stale pointers are ignored at render) |
| `created_turn` / `updated_turn` | `int` | bookkeeping |

### 2.3 Serialization round-trip

In `to_artifact_dict()` (line 397, anchor `"repair_transactions": [`), add
directly after the `repair_transactions` item:

```python
            "repair_saturation": {
                str(key): {
                    **entry,
                    "process_failures": dict(entry.get("process_failures") or {}),
                    "evidence_failed_pairs": list(entry.get("evidence_failed_pairs") or []),
                    "finalist_hashes": list(entry.get("finalist_hashes") or []),
                }
                for key, entry in self.repair_saturation.items()
            },
```

In `from_artifact_dict()` (line 456, anchor `repair_transactions=[`), add a
constructor argument directly after `repair_transactions=...`:

```python
            repair_saturation={
                str(key): _normalize_saturation_entry(value)
                for key, value in (data.get("repair_saturation") or {}).items()
                if isinstance(value, dict)
            },
```

and a module-level normalizer next to `_normalize_loaded_experiment_records`:

```python
def _normalize_saturation_entry(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_content_hash": str(value.get("candidate_content_hash") or ""),
        "attestation_key": str(value.get("attestation_key") or "none"),
        "evidence_failures": int(value.get("evidence_failures") or 0),
        "process_failures": {
            str(k): int(v)
            for k, v in (value.get("process_failures") or {}).items()
        },
        "evidence_failed_pairs": sorted(
            str(p) for p in value.get("evidence_failed_pairs") or []
        ),
        "finalist_hashes": sorted(
            str(h) for h in value.get("finalist_hashes") or []
        ),
        "readings": int(value.get("readings") or 0),
        "exhausted": bool(value.get("exhausted")),
        "pending_experiment_id": (
            str(value["pending_experiment_id"])
            if value.get("pending_experiment_id") else None
        ),
        "created_turn": int(value.get("created_turn") or 0),
        "updated_turn": int(value.get("updated_turn") or 0),
    }
```

A pre-M5.3 artifact (no `repair_saturation` key) loads as `{}` — same
missing-key convention as `repair_transactions` (see
`tests/test_investigation_state.py::…` at lines 226–235). The v2→v3 adapter
(`adapter.py`) needs no change: the dataclass default applies.

---

## 3. Failure taxonomy and classifier — `src/investigation/loop_v3.py`

### 3.1 Reason strings (complete set the repair-transaction path can emit)

**Precondition failures** — argument/binding problems detected before any
worker evaluation. They touch NO saturation counter:
`unknown_branch`, `fresh_reading_required`, `reading_branch_mismatch`,
`stale_or_unbound_reading`, `source_and_reading_already_handled`
(status `duplicate_suppressed`), and the two new blocks
`repair_saturated`, `pair_evidence_failed` (status `blocked`, §4.3).

**Process failures** — evidence was NOT adjudicated. Exactly one linked
retry per pair is permitted (`retry_of`); the retry itself does not consume
saturation budget; a SECOND process failure for the same pair is counted as
an evidence failure:

| Reason | Emitted when |
|---|---|
| `no_winner_named_with_multiple_changed_finalists` | ≥2 changed finalists and the worker named no winner, **or** named a winner that was not included in final adjudication (§4.6 check 5 — reason string mandated by master spec 406–416) |
| `unsupported_winner` | the worker named a `best_branch` that is not any recorded episode snapshot (fabricated/unnamed) |
| `winner_fork_from_failed_call` | the winner snapshot changed but NO successful composite result names it as a created fork (covers forks left behind by errored tool calls) |
| `unsupported_edit_claim` | a claimed edit string does not appear in any successful composite-action result |
| `worker_did_not_apply` | episode result `applied` is falsy |
| `install_failed` | `_dispatch_episode_install` did not return ok/deduplicated |
| `transaction_error` | outer exception guard in `_dispatch_tool` (line 1305). NOTE: this path returns a payload without appending a transaction record (pre-existing behavior, unchanged); it is therefore outside saturation accounting. Documented, accepted |
| episode-failure passthrough | `runner_error`, `schema_mismatch`, `budget_exhausted`, `cost_ceiling_reached`, `interrupted`, or an `error` string from a rejected `episode_run` — whatever lands in the existing `failure_reason or error` field |

**Evidence failures** — the pair WAS evaluated and did not support
installation. They count toward saturation and permanently retire the pair:

| Reason | Emitted when |
|---|---|
| `no_changed_finalists` | no snapshot digest differs from the source, or the worker named a real snapshot as winner while nothing changed |
| `no_op` | the worker named a REAL finalist whose content equals the source (named-but-unchanged winner), or the pre-install probe hash equals the source hash (§4.6 check 7) |
| `all_finalists_rejected` | changed finalists exist but the worker named no winner AND its own verdicts explicitly reject every changed finalist (§4.5) |
| `materially_non_improving` | deterministic collateral limit violated (check 6) or a measured host scalar strictly decreased under the default-deny policy (check 8) |

The master spec's bare evidence reason `unsupported` is never emitted: its
cases are covered by `all_finalists_rejected` / `materially_non_improving`
(worker-adjudicated-and-rejected) while fabricated names stay process
(`unsupported_winner`). The combined `ambiguous_or_unchanged_finalists`
reason is **forbidden** (master spec 306–307) and must not appear anywhere in
`loop_v3.py` after this change.

### 3.2 Classifier

Module-level in `loop_v3.py`:

```python
_EVIDENCE_FAILURE_REASONS = frozenset({
    "no_changed_finalists",
    "no_op",
    "all_finalists_rejected",
    "materially_non_improving",
})


def _classify_failure_reason(reason: Any) -> str:
    """'evidence' for the enumerated evidence reasons; 'process' otherwise.

    Default-process is deliberate: an unknown reason (new episode failure
    modes, provider error strings) must not silently consume saturation
    budget on first occurrence — the second-process-failure rule converts
    repeats into evidence failures."""
    return "evidence" if str(reason or "") in _EVIDENCE_FAILURE_REASONS else "process"
```

Every failed transaction record gains two fields:
`"failure_class"`: the classifier output for its reason, and
`"counted_evidence_failure"`: `True` iff the failure consumed saturation
budget (evidence class, or a second-or-later process failure for the pair).

---

## 4. `_dispatch_repair_transaction` rewrite — `src/investigation/loop_v3.py`

The function keeps its overall shape (lines 984–1207). Steps below are in
execution order; unchanged steps are marked.

### 4.1 Unchanged pre-checks

`unknown_branch` (988–992), reading resolution + `fresh_reading_required`
(993–1016), `reading_branch_mismatch` (1017–1025),
`stale_or_unbound_reading` (1026–1035) — unchanged.

### 4.2 Extended duplicate check (anchor: `duplicate = next(`, line 1036)

Compute first (right after the stale check):

```python
        from investigation.reading import interpretation_digest as _interp_digest
        from investigation.state import (
            attestation_key, get_or_create_saturation_entry,
            latest_attestation_for_hash, pair_digest, saturation_key,
        )

        interp_digest = _interp_digest(reading_data)
```

Then extend the duplicate predicate so identical CONTENT under a fresh
reading_id is also a duplicate:

```python
        duplicate = next(
            (
                item for item in reversed(state.repair_transactions)
                if item.get("status") == "installed"
                and item.get("source_content_hash") == source_hash
                and (
                    item.get("interpretation_id", item.get("reading_id")) == reading_id
                    or (
                        item.get("interpretation_digest")
                        and item.get("interpretation_digest") == interp_digest
                    )
                )
            ),
            None,
        )
```

(payload unchanged: `duplicate_suppressed` / `source_and_reading_already_handled`.)

### 4.3 Saturation keys + gates (new; before the attestation/note block)

Replace the current attestation selection (lines 1056–1064, anchor
`matching_attestations = [`) with the shared helper, and add the gates:

```python
        latest_attestation = latest_attestation_for_hash(
            state.verify_attestations, source_hash
        )
        att_key = attestation_key(latest_attestation)
        sat_key = saturation_key(source_hash, att_key)
        pair = pair_digest(source_hash, interp_digest)
        entry = state.repair_saturation.get(sat_key)

        if entry is not None and entry.get("exhausted"):
            return _record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn,
                payload={
                    "status": "blocked",
                    "reason": "repair_saturated",
                    "branch": branch,
                    "saturation_key": sat_key,
                    "evidence_failures": int(entry.get("evidence_failures") or 0),
                    "note": (
                        "Repair is exhausted for this candidate content and "
                        "verifier evidence. Run one alternate search/basin "
                        "experiment, compare genuinely distinct finalists, or "
                        "declare honestly unsolved."
                    ),
                },
            )
        if entry is not None and pair in (entry.get("evidence_failed_pairs") or []):
            return _record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn,
                payload={
                    "status": "blocked",
                    "reason": "pair_evidence_failed",
                    "branch": branch,
                    "pair_digest": pair,
                    "note": (
                        "This source/interpretation pair was already evidence-"
                        "evaluated and failed; it cannot be rerun under a new "
                        "name. Provide genuinely new content or evidence."
                    ),
                },
            )
        retry_of = None
        if entry is not None and int((entry.get("process_failures") or {}).get(pair, 0)) >= 1:
            retry_of = next(
                (
                    str(item.get("transaction_id") or "")
                    for item in reversed(state.repair_transactions)
                    if item.get("pair_digest") == pair
                    and item.get("status") == "failed"
                    and item.get("failure_class") == "process"
                ),
                None,
            )
```

Blocked attempts append NOTHING to `state.repair_transactions` and touch no
counter (parity with `duplicate_suppressed`).

The existing `anomalies` extraction keeps its meaning but reads from
`latest_attestation` (drop the local `matching_attestations` /
`latest_attestation` max-by-created_turn block — the helper replaces it; the
`(created_turn, episode_id)` tiebreak is a deliberate, harmless unification).

### 4.4 Episode dispatch + `base_record` (anchor: `base_record = {`, line 1086)

Episode dispatch (1074–1084) is unchanged. Extend `base_record`:

```python
        base_record = {
            "transaction_id": transaction_id,
            "source_branch": branch,
            "source_content_hash": source_hash,
            "reading_id": reading_id,                 # operational pointer into state.readings
            "interpretation_id": reading_id,          # B2 identity component (== reading_id in M5.3)
            "interpretation_digest": interp_digest,   # B2 identity component
            "attestation_key": att_key,
            "saturation_key": sat_key,
            "pair_digest": pair,
            "retry_of": retry_of,
            "episode_id": episode_payload.get("episode_id"),
            "addressed_anomalies": anomalies,
            "created_turn": turn,
        }
```

B2 note: `reading_id` remains on the record as the operational pointer (the
repair episode is still driven by the stored Reading, and mixed-vintage
artifacts carry only `reading_id`); the IDENTITY components used by the
duplicate check and saturation are `interpretation_id` /
`interpretation_digest`. In M5.3 `interpretation_id == reading_id` by
definition; M5.4 interpretation packets change the value source, not the
field names.

### 4.5 Outcome bookkeeping helper (new, nested in `run_v3`)

All failure/success paths route through one helper so counters cannot drift:

```python
    def _settle_repair_outcome(
        *, record: dict[str, Any], entry_args: tuple[str, str, str],
        changed_hashes: list[str], turn: int,
    ) -> dict[str, Any]:
        """Update the saturation entry for one FINISHED transaction record
        (failed or installed), stamp failure_class/counted_evidence_failure on
        failures, append the record to state.repair_transactions, and attach a
        compact ``saturation`` summary to the returned payload."""
        source_hash, att_key, pair = entry_args
        entry = get_or_create_saturation_entry(state, source_hash, att_key, turn)
        entry["updated_turn"] = turn
        if changed_hashes:
            entry["finalist_hashes"] = sorted(
                set(entry.get("finalist_hashes") or []) | set(changed_hashes)
            )
        if record.get("status") == "failed":
            failure_class = _classify_failure_reason(record.get("reason"))
            record["failure_class"] = failure_class
            counted = False
            if failure_class == "process":
                prior = int((entry.get("process_failures") or {}).get(pair, 0))
                entry.setdefault("process_failures", {})[pair] = prior + 1
                if prior >= 1:
                    # Second process failure for the pair counts as evidence.
                    counted = True
            else:
                counted = True
            if counted:
                entry["evidence_failures"] = int(entry.get("evidence_failures") or 0) + 1
                failed_pairs = set(entry.get("evidence_failed_pairs") or [])
                failed_pairs.add(pair)
                entry["evidence_failed_pairs"] = sorted(failed_pairs)
                if entry["evidence_failures"] >= 2:
                    entry["exhausted"] = True
            record["counted_evidence_failure"] = counted
        state.repair_transactions.append(record)
        record_with_summary = {
            **record,
            "saturation": {
                "evidence_failures": int(entry.get("evidence_failures") or 0),
                "remaining_before_exhausted": max(
                    0, 2 - int(entry.get("evidence_failures") or 0)
                ),
                "exhausted": bool(entry.get("exhausted")),
            },
        }
        return record_with_summary
```

Every existing `state.repair_transactions.append(record)` +
`_record_dispatch_result(... payload=record)` pair inside
`_dispatch_repair_transaction` is replaced by
`payload = _settle_repair_outcome(record=record, entry_args=(source_hash,
att_key, pair), changed_hashes=..., turn=turn)` followed by
`_record_dispatch_result(..., payload=payload)`. `changed_hashes` is
`sorted(changed.values())` once `changed` exists, else `[]` (the
episode-failure path at 4.4's `status != "ok"` branch passes `[]`).

The episode-failure branch (lines 1095–1104) keeps `reason =
episode_payload.get("failure_reason") or episode_payload.get("error")` and
routes through `_settle_repair_outcome` (class: process by default); its
returned tool payload keeps the existing extra key, i.e.
`{**settled_payload, "episode": episode_payload}`.

### 4.6 Winner selection + Slice-4 host acceptance (REPLACES lines 1114–1148)

Delete the current block from `result = episode_payload.get("result") or {}`
(line 1114) through the `worker_did_not_apply` return (line 1148) — in
particular the forbidden line

```python
            reason = "unsupported_winner" if requested else "ambiguous_or_unchanged_finalists"
```

must not survive in any form. Replace with the ordered checks below. Failed
checks build `record = {**base_record, "status": "failed", "reason": <reason>,
"claimed_winner": requested or None, "changed_finalists": sorted(changed),
"finalist_hashes": sorted(changed.values()), "acceptance": acceptance}` and
return via `_settle_repair_outcome`.

**Evidence extraction (module-level helper).** The evidence source is the
repair episode's own ToolCalls. They are ALREADY in memory at validation
time: `_dispatch_episode_run` extends the `episode_tool_calls` closure list
(line 757) with `EpisodeResult.tool_calls`, and every one of those ToolCalls
is stamped with the episode's id — `executor.execute` stamps
`episode_id=self.episode_id` (tools_v2 line 3165) and `execute_composite`
stamps `episode_id=getattr(executor, "episode_id", None)` (actions.py line
264). The compact `state.episode_ledger` entry does NOT retain per-tool
results (see `EpisodeResult.ledger_dict`) — and it does not need to:
validation always runs synchronously in the same dispatch that ran the
episode, and for post-hoc artifact review the full ToolCalls are merged into
`artifact.tool_calls` at finalize (loop_v3 line 1732) while the decision
itself is stored in the transaction's `acceptance` sub-record (below).
**No new ledger field is required.**

```python
def _extract_repair_evidence(tool_calls: list[Any]) -> dict[str, Any]:
    """Parse the repair episode's composite ToolCall results into the
    evidence sets the acceptance checks bind against. ``tool_calls`` is the
    episode-filtered list; each item has .tool_name and .result (JSON str).

    A result is SUCCESSFUL iff it parses to a dict with status == "ok" and no
    top-level "error" key. Failed results contribute nothing to
    supported_forks / edit_evidence (this, plus the failed_result_forks scan,
    realizes "no unresolved error invalidates a claimed edit")."""
    supported_forks: set[str] = set()
    edit_evidence: set[str] = set()
    adjudicated_sets: list[set[str]] = []
    batch_finalist_forks: set[str] = set()
    failed_result_forks: set[str] = set()
    composite_names = {
        "hypothesis_apply_reading", "hypothesis_test_word",
        "hypothesis_test_words", "branch_adjudicate",
    }
    for call in tool_calls:
        if getattr(call, "tool_name", None) not in composite_names:
            continue
        try:
            parsed = json.loads(getattr(call, "result", "") or "")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        ok = parsed.get("status") == "ok" and "error" not in parsed
        fork_fields: list[str] = []
        if isinstance(parsed.get("fork"), str):
            fork_fields.append(parsed["fork"])
        if isinstance(parsed.get("installed_fork"), str):
            fork_fields.append(parsed["installed_fork"])
        for name in parsed.get("installed") or []:
            if isinstance(name, str):
                fork_fields.append(name)
        items = [i for i in parsed.get("items") or [] if isinstance(i, dict)]
        for item in items:
            if isinstance(item.get("installed_fork"), str):
                fork_fields.append(item["installed_fork"])
        if not ok:
            failed_result_forks.update(fork_fields)
            continue
        if call.tool_name == "branch_adjudicate":
            adjudicated_sets.append({
                str(row.get("branch") or "")
                for row in parsed.get("rows") or []
                if isinstance(row, dict)
            })
            continue
        supported_forks.update(fork_fields)
        edit_evidence.update(
            str(e).strip() for e in parsed.get("edits") or []
        )
        for item in items:
            if item.get("status") == "ok":
                edit_evidence.update(
                    str(e).strip() for e in item.get("edits") or []
                )
        for finalist in parsed.get("finalists") or []:
            if isinstance(finalist, dict) and isinstance(
                finalist.get("installed_fork"), str
            ):
                batch_finalist_forks.add(finalist["installed_fork"])
    return {
        "supported_forks": supported_forks,
        "edit_evidence": edit_evidence,
        "adjudicated_sets": adjudicated_sets,
        "batch_finalist_forks": batch_finalist_forks,
        "failed_result_forks": failed_result_forks,
    }
```

Caller filters by episode:

```python
        evidence = _extract_repair_evidence([
            tc for tc in episode_tool_calls
            if getattr(tc, "episode_id", None) == episode_id
        ])
```

Also parse the worker's explicit rejections once:

```python
def _worker_rejected_targets(result: dict[str, Any]) -> set[str]:
    """Fork names the worker's own verdicts explicitly reject."""
    rejected: set[str] = set()
    for verdict in result.get("verdicts") or []:
        if not isinstance(verdict, dict):
            continue
        word = str(verdict.get("verdict") or "").strip().lower()
        if "reject" in word or word in {"discard", "discarded", "invalid", "not viable"}:
            rejected.add(str(verdict.get("target") or ""))
    return rejected
```

**Ordered checks.** `changed` is computed exactly as today (snapshot digest
!= `source_hash`). Build an ordered `acceptance_checks: list[dict]` where
each executed check appends `{"check": <name>, "passed": bool, ...detail}`;
the first failure stops the sequence and maps to a reason:

1. **`winner_named`** — resolve the winner:
   - `requested` non-empty and `requested` not among the snapshot NAMES at
     all → fail, reason `unsupported_winner` (process).
   - `requested` non-empty, is a snapshot, but not in `changed` →
     fail, reason `no_op` if its digest equals `source_hash` — it always
     does in this branch — (evidence).
   - `requested` non-empty and in `changed` → winner = requested (an explicit
     request is authoritative even if a verdict also rejects it; later checks
     still apply).
   - `requested` empty, `len(changed) == 0` → fail, `no_changed_finalists`
     (evidence).
   - `requested` empty, `len(changed) == 1` and the single changed finalist
     is NOT in `_worker_rejected_targets(result)` → winner = it (today's
     auto-select, now rejection-aware).
   - `requested` empty and every member of `changed` is in
     `_worker_rejected_targets(result)` → fail, `all_finalists_rejected`
     (evidence).
   - `requested` empty, `len(changed) >= 2`, not all rejected → fail,
     `no_winner_named_with_multiple_changed_finalists` (process).
   - `requested` non-empty, snapshots exist, `len(changed) == 0` handled by
     the second bullet; `requested` empty + `len(changed)==1` + that one
     rejected → `all_finalists_rejected` (evidence).
2. **`worker_applied`** — `bool(result.get("applied"))` else fail,
   `worker_did_not_apply` (process). (Same position as today, now after
   winner resolution as before.)
3. **`winner_fork_evidence`** — `winner in evidence["supported_forks"]` and
   `winner not in evidence["failed_result_forks"]`, else fail,
   `winner_fork_from_failed_call` (process).
4. **`edit_claims_bound`** — every claimed edit
   (`[str(e).strip() for e in result.get("edits") or []]`, exact
   case-sensitive match after strip; the labels are host-generated —
   `f"{symbol}={letter}"` from `hypothesis_apply_reading`, word-repair edit
   strings from the word tools — so the worker is expected to copy them
   verbatim) must be in `evidence["edit_evidence"]`; an empty claim list
   passes vacuously (the fork-provenance check already binds the change to a
   successful call). Else fail, `unsupported_edit_claim` (process). Record
   `{"claimed": n, "unbound": sorted(missing)}` in the check detail.
5. **`winner_adjudicated`** — only when `len(changed) >= 2`: pass iff some
   `evidence["adjudicated_sets"]` contains `winner` together with at least
   one OTHER member of `changed`, OR `winner in
   evidence["batch_finalist_forks"]` (the batch tool's deduped finalist set
   is itself a host-side adjudication). Else fail,
   `no_winner_named_with_multiple_changed_finalists` (process — reason string
   mandated by master 410–411). When `len(changed) == 1` record the check as
   passed with `{"trivial": true}`.
6. **`collateral_within_limits`** — locate the winner-producing successful
   result: the LAST successful composite result whose `fork` /
   `installed_fork` (or an item's `installed_fork`) equals `winner`; take
   its `adjudication_summary` (top-level for the singleton, item-level for
   the batch; `hypothesis_apply_reading` has none). If both
   `damaged_occurrences` and `improved_occurrences` are numeric, require
   `damaged_occurrences <= improved_occurrences`; else the check passes
   vacuously (the scalar default-deny below still guards quality). Fail →
   `materially_non_improving` (evidence). Record both numbers in the detail.
7. **`no_op_probe`** + **8. `scalar_non_decrease`** — one probe restore
   serves both. Nested helper in `run_v3`:

```python
    def _probe_snapshot_scores(
        snapshot: dict[str, Any], transaction_id: str
    ) -> tuple[str, dict[str, float | None]]:
        """Restore the winner snapshot into the live workspace under a
        reserved probe name, hash + quick-score it with the SAME renderer and
        scoring the branch cards use, and always delete the probe."""
        from investigation.state import _restore_branch_into

        probe_name = f"__repair_probe_{transaction_id}"
        snap = copy.deepcopy(snapshot)
        snap["name"] = probe_name
        _restore_branch_into(workspace, snap)
        try:
            probe_hash = _branch_hash(workspace, probe_name)
            scores = executor._compute_quick_scores(probe_name)
        finally:
            if workspace.has_branch(probe_name):
                workspace.delete(probe_name)
        return probe_hash, scores
```

   Compute `before = executor._compute_quick_scores(branch)` (the source)
   and `(probe_hash, after) = _probe_snapshot_scores(winner_snapshot,
   transaction_id)` where `winner_snapshot` is the ledger snapshot dict whose
   `name == winner`.
   - Check 7: `probe_hash != source_hash` else fail, `no_op` (evidence).
     (Winner selection already guarantees the SNAPSHOT digest differs; this
     re-checks with the exact live-workspace renderer — defensive,
     deterministic, and it doubles as the reject-before-install guarantee
     that a no-op can never install.)
   - Check 8 (default-deny): for each scalar in `("dict_rate", "quad")`
     where BOTH `before[s]` and `after[s]` are numeric, `delta = after[s] -
     before[s]`; any `delta < 0` fails → `materially_non_improving`
     (evidence). Non-measurable scalars (either side `None`) are skipped and
     recorded as `null` deltas. There is NO tolerance: the master spec's
     "small scalar decreases … only under an explicit, tested,
     ground-truth-free policy" hook is a deliberate M5.4 NON-goal here —
     represent it as a module-level constant
     `REPAIR_ACCEPTANCE_POLICY: Any = None  # M5.4 hook; None = default deny`
     checked as `if REPAIR_ACCEPTANCE_POLICY is None: <reject on any
     decrease>`. No other policy branch is implemented.

**The `acceptance` sub-record** (stored on BOTH installed and
acceptance-rejected records; master 398 + 416):

```python
        acceptance = {
            "policy": "default_deny_v1",
            "checks": acceptance_checks,           # ordered, as executed
            "supported_forks": sorted(evidence["supported_forks"]),
            "edit_evidence_count": len(evidence["edit_evidence"]),
            "adjudicated": <bool, or None when len(changed) <= 1>,
            "scores_before": before,               # {"dict_rate":…, "quad":…}
            "scores_after": after,                 # None until the probe ran
            "score_deltas": {                      # None until the probe ran
                "dict_rate_delta": <float | None>,
                "quad_delta": <float | None>,
            },
        }
```

For failures before the probe (checks 1–6), `scores_after`/`score_deltas`
are `None`. Every failed record built in this section carries
`"acceptance": acceptance` alongside `claimed_winner` /
`changed_finalists` / `finalist_hashes`, and routes through
`_settle_repair_outcome`.

### 4.7 Install + success record (anchor: `install_payload = json.loads(`, line 1150)

Only after ALL checks pass does `_dispatch_episode_install` run (unchanged
call). `install_failed` still routes through `_settle_repair_outcome`
(process). The success record keeps every existing field
(`worker_winner`, `installed_branch`, `result_content_hash`, `changed`,
`reverification_required`, `edits`, `collateral`) and adds
`"acceptance": acceptance` (with the final probe-backed scores/deltas).
The success path also routes through `_settle_repair_outcome` (it updates
`finalist_hashes` and `updated_turn`; success never touches failure
counters), replacing the bare `state.repair_transactions.append(record)` at
line 1189. The repair-agenda addressed-marking (1190–1198), evidence entry
(1199–1203) and `repair_transaction_complete` emit (1204) are unchanged;
the emitted/`_record_dispatch_result` payload is the
`_settle_repair_outcome` return value.

Scope note (deliberate): Slice-4 acceptance gates **repair_transaction
only**. The lead's direct `episode_install_branch` remains an explicit,
ungated install — it serves search/experiment adoption where a scalar dip
can be a legitimate exploration step, and the master spec's Slice 4 is
titled and scoped "Host-Validated **Repair** Acceptance".

---

## 5. Reading suppression — `_dispatch_episode_run`

Anchor: the `if kind in {"reading", "repair"}:` block (lines 717–727). After
`reading_branches` is validated to exactly one element and BEFORE
`build_candidate_reading_packet`, insert (readings only — repair-kind
episodes are already saturation-gated by their transaction):

```python
            if kind == "reading":
                from investigation.state import (
                    attestation_key, latest_attestation_for_hash, saturation_key,
                )
                reading_branch = reading_branches[0]
                content_hash = _branch_hash(workspace, reading_branch)
                att_key = attestation_key(latest_attestation_for_hash(
                    state.verify_attestations, content_hash
                ))
                sat_entry = state.repair_saturation.get(
                    saturation_key(content_hash, att_key)
                )
                if sat_entry is not None and int(sat_entry.get("readings") or 0) >= 1:
                    existing = max(
                        (
                            r for r in state.readings.values()
                            if r.get("candidate_content_hash") == content_hash
                        ),
                        key=lambda r: (
                            int(r.get("created_turn") or 0),
                            str(r.get("reading_id") or ""),
                        ),
                        default=None,
                    )
                    return json.dumps({
                        "status": "blocked",
                        "reason": "duplicate_reading_suppressed",
                        "branch": reading_branch,
                        "content_hash": content_hash,
                        "existing_reading_id": (
                            str(existing.get("reading_id")) if existing else None
                        ),
                        "note": (
                            "A reading already exists for this exact content "
                            "and verifier evidence. Reuse it via "
                            "repair_transaction, or produce new content or new "
                            "verifier evidence first."
                        ),
                    }, ensure_ascii=False)
```

Increment on success — in the reading-compile block (anchor
`if result.kind == "reading" and result.status == "ok"`, lines 809–819),
after `state.readings[reading.reading_id] = reading.to_dict()`:

```python
            from investigation.state import get_or_create_saturation_entry
            packet_hash = str(
                (inputs.get("candidate_packet") or {}).get("content_hash") or ""
            )
            if packet_hash:
                sat_entry = get_or_create_saturation_entry(
                    state, packet_hash, _reading_att_key, turn
                )
                sat_entry["readings"] = int(sat_entry.get("readings") or 0) + 1
                sat_entry["updated_turn"] = turn
```

where `_reading_att_key` is the `att_key` computed at gate time (hoist the
gate's `att_key` into a variable visible at the compile site; for
non-reading kinds it is unused). The counter caps at 1 by the gate, not by
clamping.

---

## 6. Workflow state: `repair_exhausted` — `src/investigation/context.py`

### 6.1 Menu helper (new, module-level, near `workflow_state`)

```python
def _repair_exhausted_menu(
    state: InvestigationState, branch: str | None, sat_key: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    actions = [
        (
            "Run one alternate search/basin experiment via experiment_submit; "
            "repair on this candidate and verifier evidence is exhausted."
        ),
        (
            "Compare genuinely distinct existing finalists "
            "(compare episode or branch_adjudicate)."
        ),
        (
            "Declare honestly unsolved with meta_declare_unsolved if no "
            "distinct hypothesis remains."
        ),
    ]
    pending = entry.get("pending_experiment_id")
    if pending and any(
        str(record.get("experiment_id") or "") == str(pending)
        and not record.get("collected")
        for record in state.experiment_queue
    ):
        actions.append(
            f"Collect pending experiment `{pending}` with experiment_collect; "
            "the state stays repair_exhausted until content or verifier "
            "evidence changes."
        )
    return {
        "state": "repair_exhausted",
        "branch": branch,
        "saturation_key": sat_key,
        "actions": actions,
    }


def _exhausted_entry_for(
    state: InvestigationState, branch: str | None,
    attestation: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    """The (key, entry) pair when repair is exhausted for this branch's
    current content + this attestation; None otherwise. Read-only."""
    from investigation.state import attestation_key, saturation_key

    if not branch:
        return None
    key = saturation_key(
        _branch_content_hash(state, branch), attestation_key(attestation)
    )
    entry = state.repair_saturation.get(key)
    if entry is not None and entry.get("exhausted"):
        return key, entry
    return None
```

### 6.2 Two insertion sites in `workflow_state`

Site A — repaired-branch negative attestation (anchor: the `else:` returning
`"state": "repair_required"` with `repaired_branch`, lines 98–106). Before
building that return value:

```python
            exhausted = _exhausted_entry_for(
                state, repaired_branch, repaired_attestation
            )
            if exhausted is not None:
                return _repair_exhausted_menu(
                    state, repaired_branch, exhausted[0], exhausted[1]
                )
```

Site B — best-branch partial attestation (anchor: the
`if coherence >= REPAIRABLE_COHERENCE_MIN or (` block returning
`"state": "repair_required"`, lines 120–136). Before that return:

```python
            exhausted = _exhausted_entry_for(state, best, attestation)
            if exhausted is not None:
                return _repair_exhausted_menu(state, best, exhausted[0], exhausted[1])
```

The extra `saturation_key` key in the menu dict is harmless to
`_render_workflow_state` (it reads only `state`/`branch`/`actions`) and is
what §8 uses to locate the entry.

### 6.3 Phase map + fail-closed (anchor: `by_phase = {`, lines 167–177)

```python
def allowed_episode_kinds(state: InvestigationState, executor: Any) -> list[str]:
    """Return episode kinds valid in the current workflow state."""
    phase = workflow_state(state, executor)["state"]
    by_phase = {
        "searching": ["survey", "search", "reading", "compare", "repair", "verify"],
        "candidate_reading": ["search", "reading", "compare", "repair", "verify"],
        "repair_required": ["reading", "compare", "repair", "verify"],
        "repair_exhausted": ["search", "compare", "verify"],
        "broaden_required": ["survey", "search", "compare", "verify"],
        "verified": ["compare", "verify"],
    }
    kinds = by_phase.get(str(phase))
    if kinds is None:
        warnings.warn(
            f"unknown workflow phase {phase!r}; failing closed to "
            "verify-only episode kinds",
            RuntimeWarning,
            stacklevel=2,
        )
        return ["verify"]
    return list(kinds)
```

Add `import warnings` to context.py. The `EPISODE_KINDS_FOR_CONTEXT`
constant stays (other renderer uses) but is no longer the unknown-phase
fallback. The warning mechanism is `warnings.warn(RuntimeWarning)`: visible
in logs, assertable in tests via `pytest.warns` (the repo has no global
`filterwarnings = error`, so tests must assert it explicitly — §12, T2.8).

### 6.4 `repair_required` wording fix (turns the red test green)

Site B's `repair_required` actions (anchor lines 126–136) become EXACTLY:

```python
                "actions": [
                    "Run or reuse one reading episode on the attested branch.",
                    (
                        "Run one repair_transaction with that reading; it "
                        "runs an isolated repair episode, then validates and "
                        "installs the supported changed fork."
                    ),
                    "Reverify the transaction's changed content.",
                ],
```

Site A's `repair_required` actions (anchor lines 98–106) become EXACTLY:

```python
                    "actions": [
                        "Run a fresh reading on the newly verified anomalies.",
                        (
                            "Use a new repair_transaction (an isolated repair "
                            "episode) bound to that changed content."
                        ),
                    ],
```

Both now contain the substring `repair episode`, satisfying
`test_negative_partial_attestation_creates_repair_action_menu` (which
exercises Site B). No other test asserts repair_required wording; the
`"## Workflow state: searching"` assertion and the phase-narrowing test
(`test_repair_required_state_narrows_episode_schema_and_dispatch`) are
untouched by these strings.

---

## 7. Loop gates — `src/investigation/loop_v3.py`

### 7.1 `repair_transaction` phase gate (anchor lines 1290–1300)

Unchanged code: `if phase not in {"candidate_reading", "repair_required"}:`
already blocks `repair_exhausted` (its blocked payload's `workflow_state`
field will now read `repair_exhausted`). The NEW in-dispatch gates (§4.3)
additionally cover the case where the transaction's named branch differs
from the branch `workflow_state` keyed on.

### 7.2 Live episode-kind gate (anchor lines 1249–1263)

Replace the stale turn-start check with a live one, so a transaction that
flips the state to `repair_exhausted` mid-turn also blocks a same-batch
reading/repair episode:

```python
        if name == "episode_run":
            requested_kind = str(args.get("kind") or "")
            live_kinds = set(allowed_episode_kinds(state, executor))
            if requested_kind not in live_kinds:
                return _record_dispatch_result(
                    name=name, tu=tu, turn=turn,
                    payload={
                        "status": "blocked",
                        "reason": "episode_kind_not_available",
                        "requested_kind": requested_kind,
                        "allowed_kinds": sorted(live_kinds),
                        "workflow_state": workflow_state(state, executor)["state"],
                    },
                )
```

`current_episode_kinds` remains as the turn-start set that shapes the tool
schema enum (line 1402 unchanged); only the dispatch check goes live.

---

## 8. `pending_experiment_id` wiring — `src/investigation/loop_v3.py`

In the `if name in EXPERIMENT_TOOL_NAMES:` block (anchor line 1317), after
`result_obj` is obtained and before `return json.dumps(result_obj, ...)`:

```python
            if (
                name == "experiment_submit"
                and isinstance(result_obj, dict)
                and result_obj.get("experiment_id")
            ):
                menu = workflow_state(state, executor)
                if menu.get("state") == "repair_exhausted":
                    entry = state.repair_saturation.get(
                        str(menu.get("saturation_key") or "")
                    )
                    if entry is not None:
                        entry["pending_experiment_id"] = str(
                            result_obj["experiment_id"]
                        )
                        entry["updated_turn"] = turn
```

No clearing logic: the menu validates the pointer against
`state.experiment_queue` at render time (§6.1) — once the experiment is
collected (or the record is orphaned away) the collect action simply stops
rendering; the stale pointer is inert. While the experiment is pending the
state remains `repair_exhausted` (nothing in the entry key changed) and the
menu offers collect — exactly the master 325–327 behavior. Escape happens
only through new content (e.g. `experiment_collect(install=true)` installs a
new branch whose content hash keys a fresh entry) or new verifier evidence.

---

## 9. Explicit non-goals

1. No worker-improvisable scalar-decrease policy: `REPAIR_ACCEPTANCE_POLICY
   = None` stub only (§4.6 check 8).
2. `episode_install_branch` is not acceptance-gated (§4.7 scope note).
3. `transaction_error` (outer exception guard) stays outside saturation
   accounting and does not append a transaction record (§3.1) — pre-existing
   behavior, documented.
4. No verify-episode dedup/suppression (the attestation-key seam note in
   §1.2 documents the re-verify reset loophole and the one-line mitigation).
5. No changes to episode schemas, budgets, toolsets, or `TOOLS.md` (no v2
   tool changed; `repair_transaction` is a v3 lead tool whose description is
   unchanged).
6. No `inspect_artifact.py` work (Slice 6 territory).

---

## 10. Required tests

Baseline: 1608 passed / 1 failed. After this slice: all green, including
everything below. Use the existing scripted-fake harness
(`ScriptedSession`, `VerifyWorkerFake`, `sessions_mod.register_session_builder`,
`_keyed_catton_state`) in `tests/test_loop_v3.py`; pure workflow/context
tests in `tests/test_lead_context.py`; serialization in
`tests/test_investigation_state.py`. `tests/test_episodes.py` is unchanged
(no episode-runtime behavior changes) — cited here only because its fakes
document the worker-result shapes the new loop tests reuse.

### 10.0 Existing-test updates

**U1 — `tests/test_lead_context.py::test_negative_partial_attestation_creates_repair_action_menu`**
(lines 82–99): no edit to the test; §6.4 makes it pass. Verify green.

**U2 — `tests/test_loop_v3.py::test_repair_transaction_runs_validates_installs_and_requires_reverify`**
(lines 794–941): the current fixture repairs `CATON → COTON`, which
DECREASES the host quad scalar (measured: −3.4989 → −3.8145) and would now
be default-denied. Update the fixture so the repair genuinely improves:
in `TransactionReadingWorkerFake.send`, change both `"COTON"` literals to
`"LATER"` (measured: quad −3.4989 → −3.0203, dict_rate 0.0 → 1.0 — LATER is
in the en word set); change the assertion `repaired.decryption == "COTON"`
to `== "LATER"`. Add assertions on the new record fields:

```python
    assert transactions[0]["interpretation_id"] == transactions[0]["reading_id"]
    assert len(transactions[0]["interpretation_digest"]) == 64
    assert transactions[0]["retry_of"] is None
    acceptance = transactions[0]["acceptance"]
    assert acceptance["policy"] == "default_deny_v1"
    assert [c["check"] for c in acceptance["checks"]] == [
        "winner_named", "worker_applied", "winner_fork_evidence",
        "edit_claims_bound", "winner_adjudicated", "no_op_probe",
        "scalar_non_decrease",
    ]
    assert all(c["passed"] for c in acceptance["checks"])
    assert acceptance["score_deltas"]["dict_rate_delta"] >= 0
    assert acceptance["score_deltas"]["quad_delta"] >= 0
    assert transactions[0]["installed_branch"] in acceptance["supported_forks"] or \
        transactions[0]["worker_winner"] in acceptance["supported_forks"]
```

(This test now also discharges Slice-4 acceptance bullets "a supported
singleton with bounded collateral installs and requires fresh verification"
and "acceptance records contain enough evidence for artifact review".)

`test_run_v3_reading_repair_install_adjudicate_declare_end_to_end` (COTON via
direct `episode_install_branch`) is NOT gated and stays unchanged — it also
serves as the regression that direct installs remain ungated.

### 10.1 Slice 2 tests (master 329–338, one per bullet)

Common fixture helper (add to `tests/test_loop_v3.py`): a parametrizable
`FailingRepairWorkerFake(provider, system, role)` whose submitted result is
injected per test via a module-level variable or a small factory — it must
submit `episode_submit_result` immediately with a caller-chosen
`{"applied": …, "best_branch": …, "edits": […], "verdicts": […],
"collateral": {}, "notes": "…"}` and make NO composite call (so no fork is
created ⇒ `no_changed_finalists`), or make scripted composite calls when the
test needs real forks. Reuse `TransactionReadingWorkerFake` (with per-test
reading text) for the reading side and seed attestations exactly like the
existing transaction test (episode_id `"prior_verify"` etc.).

**T2.1 `test_s2_repeated_reading_suppressed_on_unchanged_content`**
(→ "Repeated readings on unchanged content and attestation are suppressed")
Setup: `_keyed_catton_state`, seed one attestation for main's hash; lead
script: `episode_run(kind="reading", branches=["main"])` twice, then text.
Assert: first tool_result parses with a `reading_id`; second parses to
`{"status": "blocked", "reason": "duplicate_reading_suppressed",
"existing_reading_id": <first id>}`; exactly ONE reading episode in
`art.episodes`; the saturation entry has `readings == 1`.

**T2.2 `test_s2_evidence_failed_pair_blocked_under_new_as_name`**
(→ "An evidence-failed source/interpretation pair cannot be rerun under a
new `as_name`")
Setup: reading episode stores a reading; repair worker fake submits
`applied=True, best_branch=None, edits=[]` with no composite call. Lead:
reading → `repair_transaction(branch="main")` →
`repair_transaction(branch="main", as_name="second_try")`.
Assert: first transaction record `status=="failed"`,
`reason=="no_changed_finalists"`, `failure_class=="evidence"`,
`counted_evidence_failure is True`; second call's tool_result is
`{"status":"blocked","reason":"pair_evidence_failed"}`; only ONE record in
`repair_transactions`; entry `evidence_failures == 1`, `exhausted is False`.

**T2.3 `test_s2_one_process_retry_can_succeed_and_install`**
(→ "one process-failure retry can succeed and install")
Setup: repair worker fake fails FIRST transaction with
`applied=False` (after creating a real fork via `hypothesis_apply_reading`
so winner selection reaches the applied check), succeeds on the SECOND
(same reading id ⇒ same pair) exactly like the U2 flow (LATER fixture).
Assert: record 1 `reason=="worker_did_not_apply"`,
`failure_class=="process"`, `counted_evidence_failure is False`,
`retry_of is None`; record 2 `status=="installed"`,
`retry_of == record1["transaction_id"]`; entry `evidence_failures == 0`,
`process_failures[pair] == 1`, `exhausted is False`.

**T2.4 `test_s2_two_evidence_failures_enter_repair_exhausted`**
(→ "Two distinct evidence-failed repairs move the state to
`repair_exhausted`")
Setup: TWO stored readings with DIFFERENT content (e.g. reading worker fake
returns "LATER" then "WATER" on successive episodes ⇒ distinct
interpretation digests); repair worker fake always submits
`applied=True, best_branch=None`, no composite call
(`no_changed_finalists` twice). Lead: reading#1 → tx#1 (by reading_id) —
NOTE the reading-suppression gate means the second reading must be obtained
by passing `reading_id` explicitly: instead run reading once and register the
second Reading directly into `state.readings` pre-run (constructed via
`Reading.from_episode_result` with different fragment text and
`candidate_packet` bound to main's packet) — then tx#2 with that
`reading_id`. After tx#2: assert entry `evidence_failures == 2`,
`exhausted is True`; `workflow_state(state', executor')` (rebuild an
executor over `art.investigation_state` via
`InvestigationState.from_artifact_dict`) returns
`state == "repair_exhausted"` with the three §6.1 actions (assert
substrings: `"experiment_submit"`, `"genuinely distinct"`,
`"meta_declare_unsolved"`); `allowed_episode_kinds == ["search", "compare",
"verify"]`; a third `repair_transaction` in the same run is blocked with
`reason == "repair_transaction_not_ready"` and
`workflow_state == "repair_exhausted"`; an `episode_run(kind="reading")` is
blocked with `reason == "episode_kind_not_available"` and
`allowed_kinds == ["compare", "search", "verify"]`.

**T2.5 `test_s2_new_content_returns_to_candidate_reading`**
(→ "New changed content returns to `candidate_reading` and requires
verification")
Setup: take the exhausted state from a T2.4-style construction (may build
the entry directly: `state.repair_saturation[saturation_key(h, k)] =
{...exhausted entry...}` using the §1.2 helpers — direct construction is
acceptable for pure context tests); then change the best branch's content
(e.g. `state.workspace.set_mapping("main", …)` altering one letter).
Assert: `workflow_state` no longer returns `repair_exhausted`; with
`state.readings` non-empty it returns `candidate_reading` (and with an
installed-transaction record pointing at the changed branch, the Site-1
`candidate_reading` "Verify the changed candidate" menu).

**T2.6 `test_s2_saturation_roundtrip_preserves_next_action`**
(→ "Serialization/resume preserves the same next action")
In `tests/test_investigation_state.py`: build a state with one exhausted
entry (direct construction via `new_saturation_entry` + mutation) and a
matching attestation; assert
`InvestigationState.from_artifact_dict(state.to_artifact_dict())` has
`repair_saturation == state.repair_saturation` (deep equality) and that
`workflow_state(restored, executor)` equals `workflow_state(state, executor)`
exactly (same state, same actions). Also assert a legacy dict WITHOUT the
key loads to `{}` (mirror the existing `repair_transactions` missing-key
test at lines 226–235).

**T2.7 `test_s2_pending_experiment_offers_collect_and_excludes_repair_kinds`**
(→ "With an experiment pending, allowed kinds exclude reading/repair and the
menu names the pending experiment")
In `tests/test_lead_context.py`: exhausted entry with
`pending_experiment_id="exp123"`; `state.experiment_queue.append(
{"experiment_id": "exp123", "type": "automated_solver", "status": "running",
"collected": False})`. Assert menu `state == "repair_exhausted"`, some
action contains both `"exp123"` and `"experiment_collect"`;
`allowed_episode_kinds(state, ex) == ["search", "compare", "verify"]`.
Variant in the same test: with `collected: True` the collect action is
absent (the other three remain).

**T2.8 `test_s2_unknown_phase_fails_closed_to_verify_with_warning`**
Monkeypatch `investigation.context.workflow_state` to return
`{"state": "someday_phase", "branch": None, "actions": []}`; assert
`with pytest.warns(RuntimeWarning): kinds = allowed_episode_kinds(state, ex)`
and `kinds == ["verify"]`.

**T2.9 `test_s2_experiment_submit_records_pending_pointer_when_exhausted`**
(→ §8 writer) In `tests/test_loop_v3.py`: state with a directly-constructed
exhausted entry for main's (content, attestation) key + the matching seeded
attestation; monkeypatch `investigation.loop_v3.dispatch_experiment_submit`
to return `{"experiment_id": "exp999", "status": "pending"}`; lead script
calls `experiment_submit` once, then text. Assert the entry's
`pending_experiment_id == "exp999"` in `art.investigation_state
["repair_saturation"]` and that a rebuilt `workflow_state` menu's collect
action names `exp999` once a matching non-collected queue record is present.

### 10.2 Slice 4 tests (master 406–416, one per bullet)

**T4.1 `test_s4_fabricated_winner_rejected`**
(→ "A fabricated best-branch name is rejected")
Repair worker fake creates a real changed fork (one
`hypothesis_apply_reading` call, LATER-style) but submits
`best_branch="branch_i_invented"`. Assert: record `status=="failed"`,
`reason=="unsupported_winner"`, `failure_class=="process"`,
`counted_evidence_failure is False`, nothing installed
(`"transaction_repaired" not in {b.name for b in art.branches}`), and
`acceptance["checks"][0] == {"check": "winner_named", "passed": False, …}`.

**T4.2 `test_s4_fork_from_failed_call_rejected`**
(→ "A changed fork produced by a failed tool call is rejected")
Monkeypatch `investigation.actions._hypothesis_apply_reading` with a wrapper
that first forks the branch on `executor.workspace` (mimicking the partial
side effect) and then raises `RuntimeError("boom")` — `execute_composite`
converts this to `{"error": "RuntimeError: boom"}` while the fork remains in
the episode workspace (the precedent for monkeypatching `actions_mod`
attributes is `tests/test_hypothesis_actions.py` lines 548/703). Worker
submits `applied=True, best_branch=<that fork>, edits=[]`.
Assert: `reason == "winner_fork_from_failed_call"`,
`failure_class == "process"`, nothing installed, and the failing check is
`winner_fork_evidence`.

**T4.3 `test_s4_unadjudicated_multi_finalist_and_evidence_reason_split`**
(→ "An unadjudicated winner among multiple changed finalists is rejected
with process reason `no_winner_named_with_multiple_changed_finalists`;
genuinely unchanged or rejected finalist sets use distinct evidence
reasons") — a three-case matrix (three runs or three sub-blocks):
 a. worker creates TWO distinct changed forks (two `hypothesis_apply_reading`
    calls with different `as_name` + fragments), names one as `best_branch`,
    runs NO `branch_adjudicate` → `reason ==
    "no_winner_named_with_multiple_changed_finalists"`,
    `failure_class == "process"`, failing check `winner_adjudicated`;
 b. same two forks, `best_branch=None` → same reason, failing check
    `winner_named`;
 c. no composite calls, `applied=True, best_branch=None` →
    `reason == "no_changed_finalists"`, `failure_class == "evidence"`;
 d. one changed fork, `best_branch=None`, verdicts
    `[{"action": "apply_reading", "target": <fork>, "verdict": "rejected"}]`
    → `reason == "all_finalists_rejected"`, `failure_class == "evidence"`.
Also assert the literal string `"ambiguous_or_unchanged_finalists"` appears
in no record and no tool_result of any of these runs.

**T4.4 — covered by U2** (supported singleton installs + reverification +
acceptance record review).

**T4.5 `test_s4_scalar_decrease_default_denied`**
(→ default-deny + deltas stored; reuses the vacated COTON fixture)
Exactly the U2 flow but with reading text `"COTON"` (the measured
quad-decreasing repair). Assert: record `status=="failed"`,
`reason=="materially_non_improving"`, `failure_class=="evidence"`,
`counted_evidence_failure is True`; failing check `scalar_non_decrease`
with `acceptance["score_deltas"]["quad_delta"] < 0`;
`acceptance["scores_before"]`/`["scores_after"]` both present; nothing
installed; no `verify` episode ran after the transaction (the fake lead
should end with a text turn); the returned payload's
`saturation["remaining_before_exhausted"] == 1`.

**T4.6 `test_s4_no_op_named_winner_is_evidence_no_op`**
Worker creates NO fork but submits `applied=True,
best_branch="main"` (a real snapshot whose digest equals the source).
Assert `reason == "no_op"`, `failure_class == "evidence"`.

**T4.7 `test_s4_duplicate_by_interpretation_digest`** (B2 behavioral)
After one INSTALLED transaction (U2 flow), register a second Reading with
byte-identical fragments but a fresh `reading_id` into `state.readings`
bound to the same candidate hash; call
`repair_transaction(branch="main", reading_id=<new id>)`. Assert
`status == "duplicate_suppressed"`,
`reason == "source_and_reading_already_handled"`, and no second record.

### 10.3 Suite invariants

- `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` fully green
  (1608 + new tests, 0 failed).
- `grep -rn "ambiguous_or_unchanged_finalists" src/` returns nothing.
- No test may hit a paid provider; all workers are scripted fakes.

---

## 11. Local acceptance / verification (1:1 to the master spec)

| Master bullet | Where proven |
|---|---|
| S2-331 repeated readings suppressed | §5 + T2.1 |
| S2-332/333 evidence-failed pair cannot rerun under new as_name; one process retry can succeed and install | §4.3/§4.5 + T2.2, T2.3 |
| S2-334 two evidence failures → `repair_exhausted` | §4.5 latch + T2.4 |
| S2-335 new changed content → `candidate_reading` + verification | key-by-content (§1.2) + T2.5 |
| S2-336 serialization/resume preserves next action | §2.3 + T2.6 |
| S2-337/338 experiment pending: kinds exclude reading/repair; menu names it | §6.1/§6.3/§8 + T2.7 |
| S2-306 no combined `ambiguous_or_unchanged_finalists` | §4.6 check 1 + T4.3 + §10.3 grep |
| S2-322/324 explicit phase-map entry; unknown fails closed to verify with a warning | §6.3 + T2.4 (kinds), T2.8 (fail-closed) |
| S4-408 fabricated best-branch rejected | check 1 + T4.1 |
| S4-409 changed fork from failed tool call rejected | check 3 + T4.2 |
| S4-410–413 unadjudicated winner = process `no_winner_named_with_multiple_changed_finalists`; unchanged/rejected sets = distinct evidence reasons | checks 1+5 + T4.3, T4.6 |
| S4-414 supported singleton with bounded collateral installs, requires fresh verification | checks 6–8 + §4.7 + U2 |
| S4-416 acceptance records suffice for artifact review | `acceptance` sub-record (§4.6) + U2/T4.5 assertions |
| S4-389–398 full pre-install binding list | checks 1–8 in order (§4.6) |
| S4-400–404 default deny on net scalar decrease; policy not improvised | check 8 + `REPAIR_ACCEPTANCE_POLICY = None` stub + T4.5 |
| B2 generic identity names, no state migration | §1.1, §4.4, §2.2 + U2/T4.7 field assertions |
