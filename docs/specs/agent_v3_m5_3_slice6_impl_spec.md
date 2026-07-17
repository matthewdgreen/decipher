# M5.3 Slice 6 + B3 — Implementation Sub-Spec (diplomatic verification contract; C6 reversal)

Status: ready to implement. Authored 2026-07-17 against HEAD `f371bba`
(Slices 1/2/3/5+B1/B2 landed; Slice 6 and Slice 7 remain).

Authority: `docs/specs/agent_v3_m5_3_control_reliability_spec.md` ("Slice 6 —
Diplomatic Verification Contract", lines 446–500, thresholds 486–491) and
Amendment **B3** in `docs/repair_reframe_m53_comments.md` (lines 288–302).
Where this sub-spec pins a decision the master leaves open, the decision is
marked **[FIXED]** and is not the coder's to reopen. Line numbers below are
as of `f371bba`; treat them as anchors, not gospel — match on the quoted
code/docstrings.

What this slice does, in one paragraph: the text-only `verify` episode result
gains five typed fields plus an uncertainty note; **design decision C6 is
REVERSED** — `meta_declare_solution` now requires a *fresh positive*
attestation (`reader_accepts_as_solution=true` on a content-hash match);
absent, stale, weak, or negative attestations all block. `coherence` stays in
the schema as a clamped 0–10 **report-only legacy** field: it no longer gates
declaration, routing, or fallback-tier selection. The other new fields drive
context routing, fallback ordering, and repair-agenda seeding. The verifier
firewall is unchanged: candidate plaintext + generic permitted context only.

Out of scope (see §11): Slice 7 observability, `meta_declare_recovered_reading`
(M5.4), interpretation packets, any v2 gate change.

---

## 0. Baseline and blast radius

- Test baseline at `f371bba`: **1624 passed / 2 skipped**
  (`PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`).
- The verify result schema gains a **required** field, so every scripted
  verify-worker fake in the test suite that omits it will fail its episode
  (`schema_mismatch` after one retry). §12.0 enumerates every such literal.
- The routing rework interacts with two Slice-2 gates that already exist and
  must keep working:
  - the live episode-kind gate (`loop_v3.py:1749–1764`,
    reason `episode_kind_not_available`), and
  - the `repair_transaction` phase gate (`loop_v3.py:1791–1794+`,
    allowed only in `{"candidate_reading", "repair_required"}`).
  Consequence: any test that seeds a *legacy-shaped* weak attestation and then
  expects repair-flavored phases must have its seed updated with the new
  routing fields (enumerated in §12.0), because legacy records route
  conservatively to `broaden_required`.

---

## 1. `src/investigation/state.py` — record, predicate, coercers, load

### 1.1 New module constants + coercion helpers (place directly above `AttestationRecord`, ~line 111)

```python
# M5.3 Slice 6: the pre-Slice-6 positive-attestation coherence threshold.
# FROZEN migration constant — used ONLY to derive `reader_accepts_as_solution`
# for legacy serialized records (master spec 472-475). Never tune; live gating
# reads `reader_accepts_as_solution` alone.
LEGACY_DECLARE_COHERENCE = 7

DAMAGE_SCOPES = ("local", "distributed", "basin_wide")
REPAIRABILITIES = ("local_repair", "broaden", "none")


def clamp_unit_interval(value: Any) -> float:
    """Coerce a verifier 0..1 field: unparseable/NaN -> 0.0; clamp to [0, 1]."""
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return 0.0
    if coerced != coerced:  # NaN
        return 0.0
    return min(1.0, max(0.0, coerced))


def normalize_damage_scope(value: Any) -> str:
    """Out-of-enum -> conservative 'basin_wide'."""
    return value if value in DAMAGE_SCOPES else "basin_wide"


def normalize_repairability(value: Any) -> str:
    """Out-of-enum -> conservative 'none'."""
    return value if value in REPAIRABILITIES else "none"
```

### 1.2 `AttestationRecord` (anchor: `class AttestationRecord`, line 113)

Append six fields AFTER `created_turn` (all existing constructions are
keyword-only, so field order is safe):

```python
    # M5.3 Slice 6 — diplomatic verifier fields. Conservative defaults: a
    # record that never states them can neither declare nor route to repair.
    target_language_confidence: float = 0.0
    semantic_recoverability: float = 0.0
    damage_scope: str = "basin_wide"
    repairability: str = "none"
    reader_accepts_as_solution: bool = False
    uncertainty_note: str = ""
```

`to_dict()`: append the six keys at the end, in the order above
(`list(...)` not needed; plain values).

`from_dict()`: replace wholesale with:

```python
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttestationRecord":
        if "reader_accepts_as_solution" in data:
            # Strict: only a JSON true counts. Any other value (string "true",
            # 1, None) is conservative False.
            accepts_as_solution = data.get("reader_accepts_as_solution") is True
        else:
            # Legacy (pre-Slice-6) record: positive iff the prior
            # _is_positive_attestation condition held (master spec 472-475).
            accepts_as_solution = bool(data.get("reader_accepts")) and int(
                data.get("coherence") or 0
            ) >= LEGACY_DECLARE_COHERENCE
        return cls(
            branch=str(data.get("branch") or ""),
            content_hash=str(data.get("content_hash") or ""),
            renderer_id=str(data.get("renderer_id") or ""),
            episode_id=str(data.get("episode_id") or ""),
            coherence=int(data.get("coherence") or 0),
            reader_accepts=bool(data.get("reader_accepts")),
            gloss=str(data.get("gloss") or ""),
            anomalies=[str(a) for a in (data.get("anomalies") or [])],
            created_turn=int(data.get("created_turn") or 0),
            target_language_confidence=clamp_unit_interval(
                data.get("target_language_confidence")
            ),
            semantic_recoverability=clamp_unit_interval(
                data.get("semantic_recoverability")
            ),
            damage_scope=normalize_damage_scope(data.get("damage_scope")),
            repairability=normalize_repairability(data.get("repairability")),
            reader_accepts_as_solution=accepts_as_solution,
            uncertainty_note=str(data.get("uncertainty_note") or ""),
        )
```

Also update the class docstring: add one sentence — "Slice 6 adds the
diplomatic verifier fields; `reader_accepts_as_solution` alone gates
declaration (C6 reversed), `coherence` is report-only legacy."

### 1.3 The single shared positive predicate **[FIXED: unified helper]**

Place directly after `latest_attestation_for_hash` (line 233):

```python
def attestation_is_positive(attestation: dict[str, Any] | None) -> bool:
    """M5.3 Slice 6 SINGLE positive-attestation predicate (reverses C6).

    Positive == the independent reader accepts the candidate AS A SOLUTION.
    Every gating/routing/fallback consumer (AttestationPolicy, context
    workflow/hints, fallback tiering, agenda seeding, bakeoff telemetry) must
    call THIS function — the pre-Slice-6 pair (context._positive /
    loop_v3._is_positive_attestation) is deleted so the definition cannot
    drift again. Legacy dicts (no `reader_accepts_as_solution` key — e.g. a
    pre-Slice-6 artifact read raw, or a hand-seeded test record) fall back to
    the frozen pre-Slice-6 condition, mirroring AttestationRecord.from_dict.
    `coherence` appears ONLY in that legacy branch; it never gates a
    new-format record.
    """
    if not attestation:
        return False
    if "reader_accepts_as_solution" in attestation:
        return attestation.get("reader_accepts_as_solution") is True
    return bool(attestation.get("reader_accepts")) and int(
        attestation.get("coherence") or 0
    ) >= LEGACY_DECLARE_COHERENCE
```

### 1.4 Resume normalization (anchor: `from_artifact_dict`, lines 571–573)

Replace

```python
            verify_attestations=[
                dict(item) for item in data.get("verify_attestations") or []
            ],
```

with

```python
            # Slice 6: normalize every stored attestation through
            # AttestationRecord.from_dict so legacy records gain the new
            # fields (conservative defaults; positivity derived from the
            # frozen legacy condition) and resume behaves like a live run.
            verify_attestations=[
                AttestationRecord.from_dict(dict(item))
                for item in data.get("verify_attestations") or []
            ],
```

…where each element is `.to_dict()`-ed, i.e. the actual expression is
`AttestationRecord.from_dict(dict(item)).to_dict()`. Round-trip property: for
a new-format record this is the identity; for a legacy record it is a pure
field-add migration.

`to_artifact_dict` (line 508) is unchanged.

### 1.5 Explicitly UNCHANGED **[FIXED]**

- `attestation_key` (165–194): the digest fallback payload stays
  `anomalies + coherence + reader_accepts`. Adding the new fields would
  re-key existing `repair_saturation` entries on resume and silently reset
  saturation. Do not touch.
- `latest_attestation_for_hash` (220–233): unchanged; it becomes the gate's
  selection rule (§4).
- `saturation_key`, `pair_digest`, `new_saturation_entry`: unchanged.

---

## 2. `src/investigation/episodes.py` — the verify contract

### 2.1 `_VERIFY_SCHEMA` (lines 232–245) — replace wholesale

```python
# M5.3 Slice 6: the diplomatic `verify` result schema. ``coherence`` is a
# clamped 0-10 REPORT-ONLY legacy field (it no longer gates anything); the
# gate reads ``reader_accepts_as_solution`` and routing reads the
# confidence/damage fields. The local validator has no min/max support, so
# numeric ranges are stated in the contract prose and the field descriptions,
# and the dispatcher clamps out-of-range values conservatively
# (loop_v3._clamp_coherence / state.clamp_unit_interval).
# Required-set philosophy (M5.1 forensics lesson, see _READING_SCHEMA note):
# require the minimum. The FIVE legacy fields stay required (existing worker
# discipline) plus ``reader_accepts_as_solution`` — the one field that gates
# declaration must be explicit, because a silently-defaulted False is
# indistinguishable from rejection and would quietly burn the only declare
# path; a worker that omits it gets one schema-retry with the error echoed.
# The four routing fields + ``uncertainty_note`` are OPTIONAL and default
# conservatively non-positive at the dispatcher (0.0 / 0.0 / basin_wide /
# none / "").
_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "coherence": {
            "type": "integer",
            "description": "0 (gibberish) to 10 (fluent, natural text). Report-only.",
        },
        "reader_accepts": {
            "type": "boolean",
            "description": (
                "Would a fluent reader accept this as genuine, if damaged, "
                "text? (Weaker than reader_accepts_as_solution.)"
            ),
        },
        "reader_accepts_as_solution": {
            "type": "boolean",
            "description": (
                "True ONLY if the candidate, exactly as it stands, is complete "
                "enough to declare the decipherment solved. A promising but "
                "incomplete or damaged reading is false."
            ),
        },
        "target_language_confidence": {
            "type": "number",
            "description": "0.0-1.0: confidence the text is the target language at all.",
        },
        "semantic_recoverability": {
            "type": "number",
            "description": (
                "0.0-1.0: how much of the intended meaning a careful reader "
                "can recover from the candidate as written."
            ),
        },
        "damage_scope": {
            "type": "string",
            "enum": ["local", "distributed", "basin_wide"],
            "description": (
                "local = a few isolated damaged spots; distributed = damage "
                "scattered throughout; basin_wide = the text as a whole does "
                "not read as the target language."
            ),
        },
        "repairability": {
            "type": "string",
            "enum": ["local_repair", "broaden", "none"],
            "description": (
                "local_repair = a few targeted fixes would complete it; "
                "broaden = needs rework beyond spot fixes; none = no repair "
                "path is visible."
            ),
        },
        "gloss": {"type": "string"},
        "anomalies": {"type": "array", "items": {"type": "string"}},
        "uncertainty_note": {
            "type": "string",
            "description": "One or two sentences on what you are least sure of.",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "coherence", "reader_accepts", "gloss", "anomalies", "confidence",
        "reader_accepts_as_solution",
    ],
}
```

### 2.2 The verify registry entry (lines 373–389)

Budget: `EpisodeBudget(1, 1024, 90.0)` → `EpisodeBudget(1, 2048, 90.0)`.
**[FIXED]** Rationale: the required result object grew and anomalies must now
quote candidate spans; a truncated JSON submit fails the whole episode, which
is strictly worse than the extra output allowance. Wall clock and the 1-call
cap are unchanged.

Contract: replace the string wholesale (single `{language}` placeholder set,
no other braces — `_episode_system_prompt` runs `.format(language=...)` on
it):

```python
        "contract": (
            "You are a fluent reader of {language}. Below is a candidate "
            "decipherment of a historical manuscript. Judge ONLY whether it "
            "reads as real (possibly damaged or partial) {language}. Genuine "
            "historical decipherments often preserve lacunae, abbreviation "
            "scars, uncertain word boundaries, and editorial omissions — "
            "imperfection alone does not make a candidate wrong, and polish "
            "alone does not make it right. Ground every judgment in the "
            "candidate itself: quote or point to the exact spans that support "
            "it. Report: `gloss` — a clause-level paraphrase of what the text "
            "says; `anomalies` — non-words, broken syntax spans, wrong-"
            "language runs, each quoting the span it concerns; "
            "`uncertainty_note` — one or two sentences on what you are least "
            "sure of; `coherence` — an integer from 0 (gibberish) to 10 "
            "(fluent, natural {language}); `reader_accepts` — whether a "
            "fluent reader would accept it as genuine, if damaged, text; "
            "`target_language_confidence` — 0.0 to 1.0 that the text is "
            "{language} at all; `semantic_recoverability` — 0.0 to 1.0 of the "
            "intended meaning a careful reader can recover; `damage_scope` — "
            "local (a few isolated damaged spots), distributed (damage "
            "scattered throughout), or basin_wide (the text as a whole does "
            "not read as {language}); `repairability` — local_repair (a few "
            "targeted fixes would complete it), broaden (rework beyond spot "
            "fixes is needed), or none (no repair path is visible); "
            "`reader_accepts_as_solution` — true ONLY if you would accept the "
            "candidate, exactly as it stands, as a complete solved "
            "decipherment (a promising but incomplete reading is false); and "
            "`confidence` — high, medium, or low in your own verdict. You "
            "have no other information and need none — do not ask for the "
            "cipher, the key, or any score."
        ),
```

The comment above the entry (lines 367–372) keeps its content; append: "Slice
6: diplomatic multi-field verdict; the historical-imperfection framing is
generic by design (no benchmark specifics)."

### 2.3 Explicitly UNCHANGED

`_build_verify_context` (776–808): the firewall surface — candidate text +
language framing ONLY, lead goal still ignored (review F-3). No edit. The
schema/contract additions flow to the worker via `_episode_system_prompt` and
the `episode_submit_result` tool def automatically. Toolset stays empty.

---

## 3. `src/investigation/loop_v3.py` — dispatcher, digest, fallback

### 3.1 Imports / deleted local predicate

- Line 57–64: remove `DECLARE_COHERENCE` from the `investigation.context`
  import list (the constant is deleted, §5.1).
- Line 66: extend the `investigation.state` import to
  `from investigation.state import (AttestationRecord, BudgetEntry, InvestigationState, attestation_is_positive, latest_attestation_for_hash, clamp_unit_interval, normalize_damage_scope, normalize_repairability)`.
  (The in-function lazy imports at 870/1299 may stay as they are.)
- Delete `_is_positive_attestation` (lines 71–74). Its two call sites (137,
  813) switch to `attestation_is_positive`.

### 3.2 `_dispatch_verify_run` — record threading + agenda seeding + echo (lines 800–845)

Replace the success block with:

```python
        if result.status == "ok" and isinstance(result.result, dict):
            record = AttestationRecord(
                branch=branch,
                content_hash=content_hash,
                renderer_id=DECODED_TEXT_RENDERER_ID,
                episode_id=result.episode_id,
                coherence=_clamp_coherence(result.result.get("coherence")),
                reader_accepts=bool(result.result.get("reader_accepts")),
                gloss=str(result.result.get("gloss") or ""),
                anomalies=[str(a) for a in (result.result.get("anomalies") or [])],
                created_turn=turn,
                target_language_confidence=clamp_unit_interval(
                    result.result.get("target_language_confidence")
                ),
                semantic_recoverability=clamp_unit_interval(
                    result.result.get("semantic_recoverability")
                ),
                damage_scope=normalize_damage_scope(
                    result.result.get("damage_scope")
                ),
                repairability=normalize_repairability(
                    result.result.get("repairability")
                ),
                reader_accepts_as_solution=(
                    result.result.get("reader_accepts_as_solution") is True
                ),
                uncertainty_note=str(result.result.get("uncertainty_note") or ""),
            )
            record_dict = record.to_dict()
            state.verify_attestations.append(record_dict)
            # Slice 6: the agenda seeds from the verifier's REPAIRABILITY
            # verdict, not from coherence. Only a non-positive attestation
            # whose reader says targeted local fixes are worthwhile mints
            # open repair items; broaden/none verdicts route elsewhere
            # (context workflow) and must not queue local repair work.
            if (
                not attestation_is_positive(record_dict)
                and record.repairability == "local_repair"
            ):
                for anomaly in record.anomalies:
                    ...  # existing dedup + numeric_ids body UNCHANGED,
                    ...  # except the appended item dict gains two keys:
                    state.repair_agenda.append({
                        "id": max(numeric_ids, default=0) + 1,
                        "kind": "verify_anomaly",
                        "source": "verify_attestation",
                        "branch": branch,
                        "content_hash": content_hash,
                        "anomaly": anomaly,
                        "status": "open",
                        "created_turn": turn,
                        "episode_id": result.episode_id,
                        "damage_scope": record.damage_scope,
                        "repairability": record.repairability,
                    })
            payload["attestation"] = {
                "branch": branch,
                "coherence": record.coherence,
                "reader_accepts": record.reader_accepts,
                "reader_accepts_as_solution": record.reader_accepts_as_solution,
                "target_language_confidence": record.target_language_confidence,
                "semantic_recoverability": record.semantic_recoverability,
                "damage_scope": record.damage_scope,
                "repairability": record.repairability,
                "uncertainty_note": record.uncertainty_note,
                "anomalies": record.anomalies,
            }
```

`_clamp_coherence` (707–728) is UNCHANGED (coherence is still recorded,
clamped, report-only).

### 3.3 `_information_digest` (lines 654–662)

Extend the per-attestation tuple so genuinely new verifier evidence registers
as new information (all values defaulted so legacy/new records produce
uniform, sortable shapes; floats rendered as fixed-point strings for stable
JSON):

```python
            "attestations": sorted(
                (
                    str(item.get("content_hash") or ""),
                    int(item.get("coherence") or 0),
                    bool(item.get("reader_accepts")),
                    bool(item.get("reader_accepts_as_solution")),
                    f"{float(item.get('target_language_confidence') or 0.0):.4f}",
                    f"{float(item.get('semantic_recoverability') or 0.0):.4f}",
                    str(item.get("damage_scope") or ""),
                    str(item.get("repairability") or ""),
                    tuple(str(a) for a in item.get("anomalies") or []),
                )
                for item in state.verify_attestations
            ),
```

### 3.4 `_select_v3_fallback` — tier re-key (lines 124–186)

Replace the per-branch attestation selection and the tier ordering:

```python
    for name in workspace.branch_names():
        if not _active_branch(workspace, name):
            continue
        content_hash = _branch_hash(workspace, name)
        # Slice 6: the LATEST verdict on the current content governs (same
        # rule as the declare gate). An older positive superseded by a newer
        # negative on identical content does NOT qualify.
        latest = latest_attestation_for_hash(state.verify_attestations, content_hash)
        positive = latest if attestation_is_positive(latest) else None
        shortlist.append({
            "branch": name,
            "content_hash": content_hash,
            "positive_attestation": dict(positive) if positive else None,
            "scores": executor._compute_quick_scores(name),
        })

    positively_attested = [item for item in shortlist if item["positive_attestation"]]
    if positively_attested:
        # Slice 6 ordering [FIXED]: coherence no longer sorts the tier. Order
        # by the reader's meaning-recovery estimate, then language confidence,
        # then recency, then name (fully deterministic; legacy-derived
        # positives carry 0.0/0.0 and sort last).
        chosen = max(
            positively_attested,
            key=lambda item: (
                float(item["positive_attestation"].get("semantic_recoverability") or 0.0),
                float(item["positive_attestation"].get("target_language_confidence") or 0.0),
                int(item["positive_attestation"].get("created_turn") or 0),
                str(item["branch"]),
            ),
        )
        return str(chosen["branch"]), {
            "tier": "fresh_positive_attestation",
            ...  # rationale/attestation/shortlist keys unchanged
        }
```

Tier names, `fresh_compare_winner`, and `scalar_fallback` are otherwise
unchanged.

### 3.5 Fallback declaration confidence (lines 2195–2197)

Replace `self_confidence=float(...coherence...)/10.0` with:

```python
                self_confidence=round(
                    (
                        float(fallback_selection["attestation"].get(
                            "target_language_confidence") or 0.0)
                        + float(fallback_selection["attestation"].get(
                            "semantic_recoverability") or 0.0)
                    ) / 2.0,
                    4,
                ),
```

**[FIXED]** Rationale: coherence is demoted to report-only everywhere, and the
mean of the two unit-interval verdicts is a conservative, deterministic
stand-in (a legacy-derived positive yields 0.0 — acceptable: that path
requires resuming a pre-Slice-6 state into exhaustion).

### 3.6 Explicitly UNCHANGED in loop_v3

- `AttestationPolicy(attestations=state.verify_attestations)` construction
  (line 479) — same live-reference injection.
- The declaration-attach block (2117–2150): still attaches the newest
  hash-matching attestation to `sol.attestation`; since the gate now requires
  positivity, an accepted declaration always carries a positive record.
- The verify dispatch pre-checks (arity-1, existence, hash-at-dispatch,
  F2 firewall inputs) — untouched.
- `_resync_attestation_branch_on_rename`, repair-transaction identity/keying
  (1298–1418), reading suppression `_reading_att_key` — untouched
  (`attestation_key` is frozen, §1.5).
- The `repair_transaction` phase gate set `{"candidate_reading",
  "repair_required"}` (1791–1794) — untouched; routing (§5) decides which
  branches reach those phases.

---

## 4. `src/agent/tools_v2.py` — the gate (C6 reversal)

### 4.1 `AttestationPolicy` (lines 2337–2390) — replace docstring and `check_declare_solution`

New docstring:

```python
    """M5.3 Slice 6 policy (REVERSES design C6): declaration requires a fresh
    POSITIVE verify attestation.

    ``meta_declare_solution`` on branch B is allowed iff the NEWEST
    AttestationRecord whose ``content_hash`` matches B's CURRENT rendered text
    (renderer ``decoded_text_v1`` = ``_decoded_text_for_panel``) is POSITIVE
    under ``investigation.state.attestation_is_positive`` — i.e. the
    independent reader set ``reader_accepts_as_solution`` true (legacy
    records: the frozen pre-Slice-6 condition). Absent, stale, weak, and
    negative attestations ALL block. Weak fresh attestations remain routing
    evidence for the workflow (they are kept in state and drive
    repair/compare/broaden); they no longer satisfy this gate. High
    ``semantic_recoverability`` alone NEVER unlocks declaration (B3).

    Matching stays hash-primary (F11 branch-rename edge); an attestation
    recorded under the same branch name whose hash no longer matches is
    STALE. ``meta_declare_unsolved`` is NOT gated (base-class default), and
    the loop's exhaustion/error fallback bypasses the tool entirely.

    Subclasses ``NoGatesPolicy`` (not the bare base) so the v3 neutral
    finalize-phase guard is preserved; only ``check_declare_solution`` is
    overridden. The records list is the LIVE ``state.verify_attestations``
    reference (constructor-injected per the repair_agenda/finalist precedent).
    """
```

New method body:

```python
    def check_declare_solution(
        self, executor: "WorkspaceToolExecutor", args: dict
    ) -> dict[str, Any] | None:
        from agent.loop_shared import (
            _candidate_content_hash,
            _decoded_text_for_panel,
        )
        from investigation.state import (
            attestation_is_positive,
            latest_attestation_for_hash,
        )

        branch = args["branch"]
        if not executor.workspace.has_branch(branch):
            # Let the handler return its own "Branch not found" error.
            return None
        current_hash = _candidate_content_hash(
            _decoded_text_for_panel(executor.workspace, branch)
        )
        latest = latest_attestation_for_hash(self._attestations, current_hash)
        if latest is not None and attestation_is_positive(latest):
            return None  # fresh POSITIVE attestation -> declaration proceeds
        if latest is not None:
            # Fresh but weak/negative: the C6 reversal. Echo the verdict so
            # the lead can route without re-reading state.
            return {
                "status": "blocked",
                "accepted": False,
                "branch": branch,
                "reason": "attestation_not_positive",
                "attestation": {
                    "reader_accepts_as_solution": latest.get(
                        "reader_accepts_as_solution"
                    ),
                    "reader_accepts": latest.get("reader_accepts"),
                    "target_language_confidence": latest.get(
                        "target_language_confidence"
                    ),
                    "semantic_recoverability": latest.get(
                        "semantic_recoverability"
                    ),
                    "damage_scope": latest.get("damage_scope"),
                    "repairability": latest.get("repairability"),
                    "anomalies": list(latest.get("anomalies") or []),
                },
                "how": (
                    "The fresh independent reading does not accept this "
                    "candidate as a complete solution. Follow the workflow "
                    "state (repair, compare, or broaden), reverify changed "
                    "content, and declare only when the reader accepts it — "
                    "or declare honestly unsolved."
                ),
            }
        stale = any(a.get("branch") == branch for a in self._attestations)
        reason = "attestation_stale" if stale else "attestation_required"
        return {
            "status": "blocked",
            "accepted": False,
            "branch": branch,
            "reason": reason,
            "how": (
                "run a verify episode on this branch, then declare if the "
                "reader accepts it as a solution"
            ),
        }
```

Import-cycle note: the `investigation.state` import is lazy (in-method) like
the existing `agent.loop_shared` import; `investigation.state` does not
import `agent.tools_v2`, so no cycle.

### 4.2 Explicitly UNCHANGED

`DeclarationPolicy` (2267), `NoGatesPolicy` (2300), `V2GatePolicy` (2393+),
`_tool_meta_declare_solution` / `_tool_meta_declare_unsolved` (13208+), the
v2 `_reading_attestations` machinery (3022/6140–6169/13174+). v2 behavior is
byte-identical: only the v3 lead constructs `AttestationPolicy`
(loop_v3.py:479).

---

## 5. `src/investigation/context.py` — routing

### 5.1 Constants (lines 55–58)

Delete `DECLARE_COHERENCE` and `REPAIRABLE_COHERENCE_MIN`. Add:

```python
# M5.3 Slice 6 routing thresholds (master spec "Slice 6", lines 486-491).
# Host constants and calibration defaults — tunable only with paid-smoke or
# equivalent targeted evidence.
TARGET_LANGUAGE_CONFIDENCE_HIGH = 0.7
SEMANTIC_RECOVERABILITY_HIGH = 0.5
```

`LATE_VERIFY_TURNS` / `POST_ATTEST_PATIENCE` stay.

Add module import: `from investigation.state import attestation_is_positive`
(top-level; `investigation.state` does not import `investigation.context`, so
no cycle) — or keep it lazy inside the three call sites if a cycle appears at
implementation time via `InvestigationState`'s imports (it does not today:
context already imports `investigation.state.InvestigationState` at line 31,
so top-level is fine).

Delete `_positive` (lines 716–719); all three call sites (146, 178→new form,
765, 811) use `attestation_is_positive`.

### 5.2 The route function **[FIXED — completes the master's partial table]**

Add near `workflow_state`:

```python
def _attestation_route(attestation: dict[str, Any]) -> str:
    """Route a fresh NON-positive attestation (master 478-491, completed).

    Master rows: high lang confidence + high recoverability + local damage ->
    one bounded repair cycle; recognizable language + distributed damage ->
    compare/alternate search; low language confidence or basin-wide damage ->
    broaden; positive -> declare (handled by the caller). The residual
    combination the master does not enumerate (recognizable language, local
    damage, LOW recoverability) routes to compare/alternate search — per B3
    the incomplete case never routes to declaration, and low recoverability
    is no evidence that a bounded local repair can lift it. Legacy records
    (conservative defaults 0.0/0.0/basin_wide) route to broaden.
    """
    tlc = float(attestation.get("target_language_confidence") or 0.0)
    recov = float(attestation.get("semantic_recoverability") or 0.0)
    scope = str(attestation.get("damage_scope") or "basin_wide")
    if tlc < TARGET_LANGUAGE_CONFIDENCE_HIGH or scope not in {"local", "distributed"}:
        return "broaden"
    if scope == "local" and recov >= SEMANTIC_RECOVERABILITY_HIGH:
        return "repair"
    return "compare_or_search"
```

### 5.3 The attested-branch menu helper

Add (module-level, after `_exhausted_entry_for`):

```python
def _attested_menu(
    state: InvestigationState, branch: str | None,
    attestation: dict[str, Any], *, repaired: bool,
) -> dict[str, Any]:
    """Menu for a branch with a fresh NON-positive attestation (Slice 6).

    The Slice-2 exhaustion short-circuit runs FIRST for every attested route:
    an exhausted (content, evidence) pair stays `repair_exhausted` until one
    of its components changes, regardless of how the verdict fields would
    otherwise route.
    """
    exhausted = _exhausted_entry_for(state, branch, attestation)
    if exhausted is not None:
        return _repair_exhausted_menu(state, branch, exhausted[0], exhausted[1])
    route = _attestation_route(attestation)
    if route == "repair":
        if repaired:
            actions = [
                "Run a fresh reading on the newly verified anomalies.",
                (
                    "Use a new repair_transaction (an isolated repair "
                    "episode) bound to that changed content."
                ),
            ]
        else:
            actions = [
                "Run or reuse one reading episode on the attested branch.",
                (
                    "Run one repair_transaction with that reading; it "
                    "runs an isolated repair episode, then validates and "
                    "installs the supported changed fork."
                ),
                "Reverify the transaction's changed content.",
            ]
        return {"state": "repair_required", "branch": branch, "actions": actions}
    if route == "compare_or_search":
        return {
            "state": "broaden_required",
            "branch": branch,
            "actions": [
                (
                    "Compare genuinely distinct finalists "
                    "(compare episode or branch_adjudicate)."
                ),
                (
                    "Run one alternate search/basin experiment via "
                    "experiment_submit or a search episode; the verifier "
                    "reports damage local repair cannot fix — do not polish "
                    "this text."
                ),
            ],
        }
    return {
        "state": "broaden_required",
        "branch": branch,
        "actions": [
            "Reject or hold the collapsed basin.",
            "Run a different search hypothesis; do not polish this text.",
        ],
    }
```

Note the repair-route action strings are IDENTICAL to today's two
`repair_required` menus (context.py 166–172 and 198–206) — behavior-preserving
for the repair path.

### 5.4 `workflow_state` rework (lines 117–235)

Keep the overall shape; replace the attestation-driven arms:

- Transaction block (130–173): `if _positive(repaired_attestation)` →
  `if attestation_is_positive(repaired_attestation)`; the entire `else:` arm
  (156–173: exhausted-check + unconditional repair_required) becomes
  `return _attested_menu(state, repaired_branch, repaired_attestation, repaired=True)`.
  (Behavior note, intended: a post-repair verify that reports basin-wide
  damage now routes to broaden instead of another repair round.)
- Best-branch block (174–235):

```python
    attestation = _fresh_attestation(state, best) if best else None
    if attestation is not None and attestation_is_positive(attestation):
        return {  # unchanged "verified" menu
            "state": "verified",
            "branch": best,
            "actions": [
                "Declare the verified branch now.",
                "Compare only if concrete evidence identifies a distinct rival.",
            ],
        }
    if attestation is not None:
        return _attested_menu(state, best, attestation, repaired=False)
    if state.readings:
        ...  # candidate_reading, unchanged
    return ...  # searching, unchanged
```

The old coherence test (`coherence >= REPAIRABLE_COHERENCE_MIN or (gloss and
anomalies)`) and the old broaden arm (208–215) are deleted — routing is now
solely the new fields. `allowed_episode_kinds` (238–258) is UNCHANGED (phase
set is unchanged; `broaden_required` already permits `search`+`compare`,
which is where the compare/alternate route lands).

### 5.5 `workflow_hint_candidates` (lines 722–822)

- Line 765 `if _positive(attestation):` → `attestation_is_positive(...)`.
  Declare-hint text/patience unchanged.
- Lines 772–786: the repair hint's trigger becomes route-based:

```python
    elif (
        attestation is not None
        and not repair_addressed
        and _attestation_route(attestation) == "repair"
    ):
        ...  # negative_verify_repair_hint body/message UNCHANGED
```

  (Compare/broaden routes get no dedicated hint — the workflow menu carries
  those actions. **[FIXED]**)
- Line 811 `_positive(...)` → `attestation_is_positive(...)`.
- No other hint changes; `late_turn_attestation_hint` / `mid_budget` /
  `late_branch_adjudication` predicates untouched.

### 5.6 Explicitly UNCHANGED

`_fresh_attestation` (706–713), `_render_workflow_state`,
`_repair_exhausted_menu`, `_exhausted_entry_for`, the `_V3_SYSTEM_TEMPLATE`
system prompt (the structured block reason + menus carry the new contract;
prompt tuning is out of scope for this slice), `build_lead_context` assembly.

---

## 6. Analyzer + telemetry + live display

### 6.1 `scripts/inspect_artifact.py::format_attestations` (lines 351–383)

Minimal, forward-compatible change (Slice 7 adds more sections; do not
restructure the file). Add at the top of the module's imports (src is already
on `sys.path`): `from investigation.state import attestation_is_positive`.
Replace the table body:

- Docstring: "…Slice 6: adds a verdict column (positive/weak/negative) and
  the diplomatic verifier fields; legacy records classify via the frozen
  legacy rule and render n/a for absent fields."
- Verdict rule **[FIXED]**: `"positive"` if `attestation_is_positive(a)`;
  else `"weak"` if `a.get("reader_accepts")` is truthy; else `"negative"`.
- Header/row (drop the old `accepts` column — verdict subsumes it; keep the
  `*declared` marker logic byte-identical):

```python
    lines.append(
        f"  {'branch':<18} {'verdict':<9} {'lang':>5} {'recov':>5} "
        f"{'scope':<11} {'repair':<13} {'coher':>5} {'anoms':>5}  gloss"
    )
    ...
        def _unit(key: str) -> str:
            v = a.get(key)
            return f"{float(v):.2f}" if isinstance(v, (int, float)) else "n/a"
        lang = _unit("target_language_confidence")
        recov = _unit("semantic_recoverability")
        scope = str(a.get("damage_scope") or "n/a")
        repair = str(a.get("repairability") or "n/a")
        gloss = str(a.get("gloss") or "").replace("\n", " ")[:36]
        lines.append(
            f"  {branch:<18} {verdict:<9} {lang:>5} {recov:>5} {scope:<11} "
            f"{repair:<13} {coher_s:>5} {anoms:>5}  {gloss}{marker}"
        )
```

`uncertainty_note` is NOT added to the table (available in raw JSON)
**[FIXED]**.

### 6.2 `scripts/run_v3_bakeoff.py` (lines 316–319)

Delete the local `_is_positive_attestation` def; immediately after the
existing `sys.path.insert(0, str(REPO_ROOT / "src"))` (line 41) add:

```python
from investigation.state import attestation_is_positive as _is_positive_attestation  # noqa: E402
```

Call sites (342–343) unchanged. Raw legacy artifact dicts keep classifying
identically via the predicate's legacy fallback; new-format artifacts
classify by `reader_accepts_as_solution`. `_attestation_sort_key` and the
telemetry row fields are unchanged (Slice 7 owns any telemetry expansion).

### 6.3 `src/agent/narrate.py::_on_declared_solution` (lines 364–377)

Replace the accept-word derivation only:

```python
            if "reader_accepts_as_solution" in attestation:
                accept_word = (
                    "reader accepts as solution"
                    if attestation.get("reader_accepts_as_solution")
                    else "reader does not accept as solution"
                )
            else:
                accepts = attestation.get("reader_accepts")
                accept_word = "reader accepts" if accepts else "reader rejects"
```

Line format otherwise unchanged (legacy payloads — e.g.
tests/test_cli_observability.py:67 — render byte-identically).

---

## 7. Documentation edits (required by repo policy)

### 7.1 `TOOLS.md` (lines ~1578–1594, the "verify (M5)" section)

- Result list: replace "`{coherence (0–10), reader_accepts, gloss, anomalies,
  confidence}`" with "`{coherence (0–10, report-only), reader_accepts,
  reader_accepts_as_solution, target_language_confidence (0–1),
  semantic_recoverability (0–1), damage_scope
  (local|distributed|basin_wide), repairability (local_repair|broaden|none),
  gloss, anomalies, uncertainty_note, confidence}`".
- Gate paragraph: replace the "A weak attestation … does NOT block …" sentence
  and the reason list with: "`meta_declare_solution` is allowed only when the
  NEWEST attestation matching the branch's current rendered text is POSITIVE
  (`reader_accepts_as_solution=true`; M5.3 Slice 6, reversing C6). A fresh
  weak/negative attestation blocks with `reason: attestation_not_positive`
  and echoes the verdict fields; a missing or hash-mismatched attestation
  blocks with `reason: attestation_required | attestation_stale`. Weak
  attestations remain routing evidence for the workflow."

### 7.2 `CLAUDE.md` — the V3 "M5" bullet

Replace "**M5** — verification-gated declaration: `meta_declare_solution` is
unblocked only by a `verify` episode attestation (or a strong
`meta_attest_reading_comprehensibility` score) whose content hash matches the
branch's current text." with "**M5/M5.3** — verification-gated declaration:
`meta_declare_solution` is unblocked only by a fresh `verify`-episode
attestation whose content hash matches the branch's current text AND whose
`reader_accepts_as_solution` is true (Slice 6 reversed C6; weak attestations
route repair/compare/broaden instead)."

---

## 8. The 7-consumer migration (master 469–472), each with its edit

| # | Consumer | File:anchor | Change | Why |
|---|---|---|---|---|
| 1 | `AttestationPolicy` | `src/agent/tools_v2.py:2337–2390` | §4.1: gate = newest hash-matching attestation must be positive; new block reason `attestation_not_positive` with verdict echo; docstring rewritten (it documented C6 verbatim) | The C6 reversal itself |
| 2 | Positive-attestation / fallback tiering | `src/investigation/loop_v3.py:71–74` (delete), `124–186` (§3.4), `2195–2197` (§3.5) | Shared predicate; tier re-keys on `reader_accepts_as_solution` via latest-verdict-governs; deterministic (recoverability, language-confidence, turn, name) ordering; confidence formula off coherence | Master: "the fallback tier `fresh_positive_attestation` re-keys on the same new positive condition"; coherence must not select tiers |
| 3 | Context routing | `src/investigation/context.py:55–58, 117–235, 716–719, 722–822` (§5) | Constants 0.7/0.5; `_attestation_route` + `_attested_menu` mapped onto EXISTING phases; exhaustion short-circuit first; coherence constants deleted | Master routing table 478–491; no new phases; Slice-2 stickiness preserved |
| 4 | Repair-agenda seeding | `src/investigation/loop_v3.py:813–839` (§3.2) | Seed only non-positive + `repairability == "local_repair"`; items carry `damage_scope`/`repairability` | Agenda seeds from the verifier's repair verdict, not coherence |
| 5 | State serialization / resume defaults | `src/investigation/state.py:111–162, 571–573` (§1) | Six new fields; `from_dict` legacy derivation (positive iff `reader_accepts` and `coherence>=7`); conservative defaults 0.0/0.0/basin_wide/none; load normalization | Master 472–475 legacy rule; resume round-trip |
| 6 | Analyzer output | `scripts/inspect_artifact.py:351–383`, `scripts/run_v3_bakeoff.py:316–319`, `src/agent/narrate.py:364–377` (§6) | Verdict column + new fields, minimal + forward-compatible (Slice 7 adds sections); bakeoff predicate deduplicated onto the shared helper; narrate accept-word | A weak-but-declared (legacy) vs positive-declared solve must be visibly different offline and live |
| 7 | Verify contract | `src/investigation/episodes.py:232–245, 373–389` (§2) | Schema + required set (+`reader_accepts_as_solution` only), contract text with historical-imperfection framing + quote-the-evidence, budget 1024→2048 output tokens | Master 448–463 |

---

## 9. B3 boundary (P2) — enforced structurally

- `reader_accepts_as_solution` answers EXACTLY "complete enough to declare
  **solved**". The only code paths that can produce a solved/declared outcome
  are (a) `AttestationPolicy.check_declare_solution` returning `None` and
  (b) the `fresh_positive_attestation` fallback tier — both read ONLY
  `attestation_is_positive`. No consumer reads `semantic_recoverability`,
  `target_language_confidence`, `damage_scope`, `repairability`, or
  `coherence` on the way to a solved status; they appear only in routing,
  ordering among already-positive candidates, seeding, and reports.
- The high-recoverability/incomplete case (`reader_accepts_as_solution=false`,
  high `semantic_recoverability`) routes to `repair_required` (one bounded
  cycle, Slice-2 saturation still applies) when damage is local, else to
  `broaden_required` (compare/alternate-search actions) — never to `verified`
  and never past the gate (§5.2; test §12.1-B3).
- `meta_declare_recovered_reading`, composite attestation hashes, and any
  recovered-reading accounting are M5.4 and MUST NOT be built, stubbed, or
  named in code in this slice.

---

## 10. Deliberate small decisions (so nothing is reopened)

1. Gate selection rule: the **newest** hash-matching attestation governs
   (`latest_attestation_for_hash`), not "any positive match" — the latest
   verdict on identical content wins. Same rule re-used in the fallback tier.
2. Strict positivity on raw values: `... is True` (a string `"true"` or `1`
   is conservative False); `from_dict` normalizes with the same strictness.
3. `attestation_key` digest fallback is FROZEN (§1.5) — no saturation re-key.
4. Legacy records route to `broaden` (conservative defaults). Consequence: a
   pre-Slice-6 serialized state resumed mid-repair re-routes its weak
   candidate to `broaden_required` unless/until fresh verifier evidence
   arrives. Accepted and documented (master mandates conservative legacy
   loads; only positivity has a legacy derivation).
5. The verify episode's `max_output_tokens` rises 1024→2048; call cap and
   wall clock unchanged.
6. `uncertainty_note` is the master's "uncertainty note" field name; optional
   in the schema; empty-string default.
7. The existing `confidence` (high/medium/low) field stays, stays required,
   stays report-only.
8. No system-prompt (`_V3_SYSTEM_TEMPLATE`) edit in this slice.
9. `_information_digest` gains the new fields (§3.3) so a changed verdict on
   unchanged text is "new information".
10. `run_v3_bakeoff` imports the shared predicate rather than keeping a
    drift-prone local copy.

## 11. Explicit non-goals

- Slice 7 (workspace-snapshot branch distinctions, analyzer sections for
  budgets/suppressions/repair-cycles/saturation) — separate slice; §6.1 is
  deliberately minimal.
- `meta_declare_recovered_reading` / M5.4 Part C items (see §9).
- Any change to `V2GatePolicy`, the v2 loop, `_reading_attestations`, or the
  v2 declare cascade.
- Any change to `_build_verify_context` (firewall surface), episode
  isolation, or the experiment queue.
- Threshold tuning: 0.7 / 0.5 land exactly as the master states.

---

## 12. Required tests

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`. Baseline
1624 passed / 2 skipped; the suite must end green with the changes below.

### 12.0 Existing-test updates (complete enumeration)

**Behavioral flips (the C6 reversal):**

- `tests/test_verify_attestation.py::test_attestation_policy_weak_allows_and_records_weakness`
  (line 196) → REPLACE with `test_attestation_policy_weak_blocks_declaration`:
  same weak fixture (`coherence=2, reader_accepts=False`, plus
  `reader_accepts_as_solution=False`); assert `status == "blocked"`,
  `reason == "attestation_not_positive"`, `executor.solution is None`,
  `executor.terminated is False`, and the block payload's `attestation` echo
  carries `reader_accepts_as_solution is False` and the two anomalies.
- `tests/test_loop_v3.py::test_run_v3_weak_attestation_allows_declare_and_carries_weakness`
  (line 420) → REPLACE with
  `test_run_v3_weak_attestation_blocks_declare_and_seeds_agenda`:
  `_WeakVerifyFake` result becomes `{"coherence": 3, "reader_accepts": False,
  "reader_accepts_as_solution": False, "target_language_confidence": 0.8,
  "semantic_recoverability": 0.7, "damage_scope": "local",
  "repairability": "local_repair", "uncertainty_note": "middle clause",
  "gloss": ..., "anomalies": ["non-word run", "broken clause"],
  "confidence": "low"}`. Assert: every `meta_declare_solution` result has
  `reason == "attestation_not_positive"`; `art.status == "unsolved"`;
  `art.solution is None`; `art.attestations[0]["reader_accepts_as_solution"] is False`;
  the repair agenda contains exactly the two anomalies with
  `source == "verify_attestation"` and `repairability == "local_repair"`.
- `tests/test_loop_v3.py::test_run_v3_out_of_scale_coherence_records_floor_not_maximum`
  (line 462): keep the coherence-floor asserts (`att["coherence"] == 0`,
  `reader_accepts is False`); the tail FLIPS — the fake gains
  `"reader_accepts_as_solution": False` (+ any-valid routing fields), and the
  final asserts become `art.status == "unsolved"` and `art.solution is None`
  (the hash-matched weak attestation no longer lets the declaration through).
- `tests/test_loop_v3.py::test_run_v3_positive_attestation_drives_attested_fallback`
  (line 336): `VerifyWorkerFake` (line 88) result becomes positive-new-format
  (below); the `self_confidence == 0.9` assert becomes
  `abs(art.solution.self_confidence - 0.85) < 1e-9` ((0.9+0.8)/2 from the
  fake); also assert
  `art.fallback_selection["attestation"]["reader_accepts_as_solution"] is True`.
- `tests/test_lead_context.py::test_negative_partial_attestation_creates_repair_action_menu`
  (line 82): the seeded dict gains `"reader_accepts_as_solution": False,
  "target_language_confidence": 0.8, "semantic_recoverability": 0.7,
  "damage_scope": "local", "repairability": "local_repair"` so it still
  exercises `repair_required`. ADD companion
  `test_legacy_attestation_routes_to_broaden`: the ORIGINAL old-shape dict →
  `workflow_state(...)["state"] == "broaden_required"` (documents the
  conservative legacy routing, decision §10.4).
- `tests/test_loop_v3.py::test_repair_required_state_narrows_episode_schema_and_dispatch`
  (line 285): the seeded dict (291–299) gains the same five routing fields as
  above so the phase stays `repair_required` and the kind-narrowing asserts
  hold.

**Scripted verify results that must gain the new REQUIRED field (episodes
fail `schema_mismatch` otherwise) — add `"reader_accepts_as_solution": <bool>`
(and, where the test drives routing, the four routing fields):**

- `tests/test_loop_v3.py::VerifyWorkerFake.send` (line 88) — becomes
  `{"coherence": 9, "reader_accepts": True, "reader_accepts_as_solution": True,
  "target_language_confidence": 0.9, "semantic_recoverability": 0.8,
  "damage_scope": "local", "repairability": "local_repair",
  "uncertainty_note": "", "gloss": "reads as clear English", "anomalies": [],
  "confidence": "high"}`. (This fake gates most solve-path tests: 145, 177,
  190, 336, 486, 532, 547, 751, 898, and the s2/s4 tests using the fixture.)
- `tests/test_loop_v3.py::_WeakVerifyFake.send` (line 411) — per the flip above.
- `tests/test_loop_v3.py::_OutOfScaleVerifyFake.send` (line 453) — add
  `"reader_accepts_as_solution": False` (+ valid routing fields, e.g.
  basin_wide/none/0.2/0.1).
- `tests/test_verify_attestation.py` result literals at lines 296 and 313
  (`test_verify_episode_zero_tool_submit`, `test_verify_episode_budget_tagged`)
  — add `True` / `False` respectively (the `res.result == good` assert keeps
  holding since submit echoes the object).
- `tests/test_episodes.py` literals at lines 498 (`verify_good`), 881, 912.
- `tests/test_experiments.py:742`.
- `tests/test_ground_truth_firewall.py:729–733` (the inline `_VerifyFake`) —
  add `"reader_accepts_as_solution": False`; the firewall assertions are
  otherwise untouched and MUST stay green (test (g)).
- `tests/test_m6_m5_note_fixes.py:266` (the F7-area fake) — add `True` +
  positive routing fields.

**Hand-seeded state attestations that must gain routing fields (because the
Slice-2 phase gates would otherwise block the flow under test):**

- `tests/test_loop_v3.py:907–912`
  (`test_repair_transaction_runs_validates_installs_and_requires_reverify`) —
  add `False`/0.8/0.7/local/local_repair.
- `tests/test_loop_v3.py::_seed_negative_attestation` (line 1188) — same
  five fields, once, covering all s2/s4 flow tests that repair.

**Deliberately UNCHANGED seeds (they become pins):**

- `tests/test_m6_m5_note_fixes.py:113–118`
  (`test_f5_rename_resyncs_attestation_branch_no_mislabel`): the old-shape
  positive seed (`reader_accepts=True, coherence=9`) now pins the LEGACY
  positive derivation at the gate — do not modernize it.
- `tests/test_lead_context.py::_seed_exhausted` (line 376) and
  `tests/test_investigation_state.py:325–329`: old-shape seeds keep returning
  `repair_exhausted` because the exhaustion short-circuit precedes routing —
  do not modernize; they pin exhaustion-first.
- `tests/test_lead_context.py::test_late_turn_attestation_hint` (line 332)
  and `tests/test_cli_observability.py:67`: unchanged, still green.
- `tests/test_run_v3_bakeoff.py` (157–222): unchanged, still green via the
  legacy fallback.

### 12.1 New tests

**tests/test_verify_attestation.py** (the declaration-policy file):

- `test_attestation_policy_positive_match_allows_and_accepts` — rename/adjust
  of the current match test: `_attestation_for` helper (line 69) gains
  keyword args `reader_accepts_as_solution=True,
  target_language_confidence=0.9, semantic_recoverability=0.8,
  damage_scope="local", repairability="local_repair", uncertainty_note=""`
  (positive-shaped defaults); a positive attestation → declaration `ok`,
  `executor.terminated is True`. (Covers required test (a).)
- `test_attestation_policy_high_recoverability_alone_does_not_unlock` (B3):
  attestation with `semantic_recoverability=1.0, target_language_confidence=1.0,
  damage_scope="local", repairability="local_repair", reader_accepts=True,
  coherence=10` but `reader_accepts_as_solution=False` → blocked,
  `reason == "attestation_not_positive"`.
- `test_attestation_policy_latest_verdict_governs`: two attestations on the
  SAME current hash — older positive (`created_turn=1`), newer negative
  (`created_turn=2`) → blocked; reversed order → allowed.
- `test_attestation_policy_legacy_record_positive_via_old_condition`:
  old-shape dict (no new keys) `reader_accepts=True, coherence=7` → allowed;
  `coherence=6` → blocked. (Pins §1.3's legacy branch at the gate.)
- Extend `test_attestation_record_from_dict_defaults` (line 340): new-field
  conservative defaults on `{"content_hash": "abc"}` —
  `target_language_confidence == 0.0`, `semantic_recoverability == 0.0`,
  `damage_scope == "basin_wide"`, `repairability == "none"`,
  `reader_accepts_as_solution is False`, `uncertainty_note == ""`.
- `test_attestation_record_legacy_load_derivation` (required test (d)):
  `from_dict({"reader_accepts": True, "coherence": 9})` →
  `reader_accepts_as_solution is True`; with `coherence=6` or
  `reader_accepts=False` → False; explicit
  `{"reader_accepts_as_solution": False, "reader_accepts": True,
  "coherence": 10}` → False (the key, when present, wins); clamping —
  `from_dict({"reader_accepts_as_solution": True,
  "target_language_confidence": 1.7, "semantic_recoverability": -0.2,
  "damage_scope": "LOCAL", "repairability": "spolish"})` → 1.0 / 0.0 /
  "basin_wide" / "none".
- Keep `test_attestation_policy_absent_blocks`,
  `test_attestation_stale_on_key_rendered_branch`,
  `test_no_attestation_blocks_branch_that_passes_v2_gates`,
  `test_declare_unsolved_not_gated_by_attestation` green unmodified except
  the helper default change (covers required test (c)).

**tests/test_lead_context.py** (routing table, required test (e)):

- `test_slice6_routing_table` (parametrize over
  `(tlc, recov, scope, accepts_solution, expected_state, action_marker)`):
  - `(0.9, 0.8, "local", True)` → `verified`
  - `(0.9, 0.8, "local", False)` → `repair_required`
  - `(0.9, 0.8, "distributed", False)` → `broaden_required` with first action
    containing "Compare genuinely distinct finalists"
  - `(0.9, 0.2, "local", False)` → `broaden_required` (compare variant — the
    completed residual row)
  - `(0.3, 0.9, "local", False)` → `broaden_required` with action containing
    "Reject or hold"
  - `(0.9, 0.9, "basin_wide", False)` → `broaden_required` ("Reject or hold")
  Each case seeds one fresh attestation on `main` via the content hash (same
  scaffolding as line 82) and asserts `workflow_state(...)`.
- `test_slice6_repair_route_exhaustion_short_circuits`: same repair-route
  attestation + an exhausted saturation entry keyed on (hash,
  `attestation_key(att)`) → `repair_exhausted`.
- `test_negative_verify_repair_hint_only_for_repair_route`: a
  distributed-damage non-positive attestation on the best branch → no
  `negative_verify_repair_hint` in `workflow_hint_candidates`; a local/high
  one → hint present.
- `test_legacy_attestation_routes_to_broaden` (from §12.0).

**tests/test_loop_v3.py** (fallback re-key, required test (f); dispatcher
coercion):

- `test_v3_fallback_rekeys_on_reader_accepts_as_solution`: state with two
  active branches of distinct content; branch A's latest matching attestation
  positive with `semantic_recoverability=0.6`, branch B's positive with
  `0.9` → `_select_v3_fallback` returns B, tier
  `fresh_positive_attestation`; then append a NEWER negative attestation for
  B's hash → returns A; make A's newest negative too → tier falls through to
  `fresh_compare_winner`/`scalar_fallback` (no positive tier).
- `test_run_v3_verify_dispatcher_clamps_and_defaults`: a verify fake
  submitting `target_language_confidence=1.7, semantic_recoverability=-0.2`
  and OMITTING `damage_scope`/`repairability`/`uncertainty_note` (schema-valid:
  they are optional) → recorded attestation has 1.0 / 0.0 / "basin_wide" /
  "none" / ""; agenda NOT seeded (repairability "none") even though
  anomalies are present and the record is non-positive.

**tests/test_investigation_state.py**:

- `test_attestations_normalized_on_slice6_load` (required test (d), resume
  side): serialize a state, strip the six new keys from the stored
  attestation dict (simulating a pre-Slice-6 artifact), reload via
  `from_artifact_dict` → the record carries derived
  `reader_accepts_as_solution` (True for accepts+coherence≥7 seed, False
  otherwise) and conservative routing defaults; a second dump/load round-trip
  is byte-stable (`to_artifact_dict` equality).

**tests/test_episodes.py**:

- `test_verify_schema_requires_reader_accepts_as_solution`: a verify worker
  submitting the OLD five-field result twice → first submit gets the
  schema-retry tool_result (`schema_errors` mentions
  `reader_accepts_as_solution`), second identical submit →
  `status == "episode_failed"`, `failure_reason == "schema_mismatch"`.
- `test_verify_contract_mentions_historical_imperfections`:
  `_episode_system_prompt(verify_spec, "en")` contains "lacunae",
  "abbreviation scars", "uncertain word boundaries", "editorial omissions",
  "quote", and "reader_accepts_as_solution"; and still contains the
  no-other-information sentence.

**tests/test_ground_truth_firewall.py** (required test (g)):

- Existing `test_v3_verify_episode_never_sees_ground_truth_or_scores` stays
  green after the fake-result update — it already asserts the rendered
  context + system prompt + sent blocks are GT-free and score-free, which now
  covers the new contract text and schema automatically. No new firewall test
  needed; do NOT weaken any assertion.

**tests/test_inspect_artifact.py**:

- `test_format_attestations_shows_verdict_and_new_fields`: artifact dict with
  one new-format positive record and one legacy weak record
  (`reader_accepts=True, coherence=4`, no new keys) → output contains
  `positive` and `weak` rows; legacy row renders `n/a` for lang/recov and the
  declared marker logic still works.

**tests/test_run_v3_bakeoff.py**:

- `test_positive_attestation_available_respects_new_field`: attestation
  `{"reader_accepts": True, "coherence": 9,
  "reader_accepts_as_solution": False}` →
  `positive_attestation_available is False`; with `True` → True; legacy dict
  `{"reader_accepts": True, "coherence": 8}` → True (fallback).

### 12.2 Suite invariants

- Full suite green; no remaining reference to `DECLARE_COHERENCE`,
  `REPAIRABLE_COHERENCE_MIN`, `_is_positive_attestation` (loop_v3), or
  `_positive` (context) anywhere in `src/` or `scripts/`
  (`grep -rn` in review).
- No new module-level import cycles (`python -c "import investigation.loop_v3, agent.tools_v2, investigation.context"` under `PYTHONPATH=src`).
- v2 pin: the existing V2GatePolicy/declare-cascade tests in
  `tests/test_agent_reliability.py` and `tests/test_episode_pins.py` pass
  unmodified.

---

## 13. Local acceptance / verification (1:1 to master Slice 6, lines 446–500)

| Master requirement | Where satisfied |
|---|---|
| R1 (448–455): result fields `target_language_confidence`, `semantic_recoverability`, `damage_scope`, `repairability`, `reader_accepts_as_solution`, gloss/anomalies/uncertainty note | §2.1 schema, §1.2 record, §3.2 threading; tests §12.1 (episodes + dispatcher) |
| R2 (457–460): verifier sees only canonical key-derived candidate + generic context; no plaintext/key/score/alignment/editorial/accuracy | §2.3 (`_build_verify_context` untouched), firewall test §12.1(g) |
| R3 (460–463): instructions cover lacunae/abbreviation scars/uncertain boundaries/editorial omissions; must quote/point to candidate evidence | §2.2 contract text; `test_verify_contract_mentions_historical_imperfections` |
| R4 (465–468): C6 reversed — fresh positive `reader_accepts_as_solution=true` required; absent/stale/weak/negative cannot satisfy the gate | §4.1; tests (a)(b)(c) in §12.0/§12.1 |
| R5 (468–469): weak fresh attestations remain routing evidence | §3.2 (records kept + agenda), §5 (routing reads them); weak-blocks test asserts the record persists in `art.attestations` |
| R6 (469–472): schema ships together with the 7 consumer migrations | §8 table, §§1–7 |
| R7 (472–475): legacy loads conservatively; positive iff `reader_accepts` ∧ `coherence>=7` | §1.2/§1.3/§1.4; tests (d) |
| R8 (475–476): `fresh_positive_attestation` tier re-keys on the new condition | §3.4; test (f) |
| R9 (478–484): routing rows repair / compare-or-alternate-search / broaden / declare-promptly | §5.2–§5.4 mapped onto existing phases; routing-table test (e) |
| R10 (486–489): thresholds 0.7 / 0.5 as host constants, tunable only with paid-smoke evidence | §5.1 constants + comment |
| R11 (489–491): `coherence` stays clamped 0–10 report-only; gates nothing | `_clamp_coherence` untouched (§3.2); every live gate/route/sort reads the new fields (§§3–5); out-of-scale test keeps the floor behavior |
| R12 (493–495): "likely correct basin with recoverable meaning, but not a complete solution" is expressible without declaring solved | weak attestation carries tlc/sr/scope/repairability in state, artifact, analyzer (§6), while gate blocks (§4); B3 tests |
| R13 (497–500) + B3: solved gate not overloaded; recovered-reading terminal deferred | §9, §11; `test_attestation_policy_high_recoverability_alone_does_not_unlock` |
