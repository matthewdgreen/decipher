# Declaration hardening — make correct solves declarable (P1–P4)

**Status:** spec (2026-07-19). Implements workplan step 1, extended by the
2026-07-19 agentic frontier findings (`docs/evidence/agentic_frontier_suite_results.md`,
finding 3: the "solve succeeds, declaration blocked" class).

## 0. Problem statement and evidence

Three independent defects share one shape: the solver produces a **correct**
(or near-perfect) plaintext, but the branch the agent can verify/declare does
not carry that plaintext, so the verification-gated declaration path (M5.3 C6)
correctly refuses — a **solved cipher records as unsolved**. A fourth item is a
routing bug surfaced by the same run.

| id | evidence | defect |
|---|---|---|
| P1 | frontier fs5 (investigation `eab74802ecec`): automated Vigenère experiment recovered key `IHBMEP` + the FULL plaintext, but `decode_show` rendered `???…` and the reader rejected the branch | the **default experiment installer** installs only the flat substitution key; periodic/transposition results have an intentionally empty flat key and their plaintext in `final_decryption`, which the installer ignores |
| P2 | round-4 composite dogfood (memory `composite-repair-padding-todo` B): a PERFECT crack carries the columnar padding tail `MMMM`; verifier scored 9/10 but the exact-content gate refused the non-language tail | the composite result offers **no padding-trimmed candidate**, so no declarable branch exists for a padded columnar composite |
| P3 | frontier fs7 (investigation `43ca1ea1f3c3`): an `automated_solver` experiment labeled `cipher_system=bifid` was hijacked by the Slice-B content auto-route into the composite sub+transposition peel | the **content auto-route ignores an explicit fractionation-family label**; bifid is not a sub+transposition composite and the peel cannot solve it |
| P4 | Codex dogfood (memory `composite-repair-padding-todo` A): "word repair could not operate because composite branches expose no ordinary base key" | repair surfaces give a **confusing failure** on decoded-text (slim-record) branches; the limitation must be surfaced as a structured, named error (fixing repair itself is out of scope) |

Out of scope (documented, deliberately): a composite/decoded-branch repair
primitive (interpretation-packet direction, M5.4-adjacent); the fs6/fs4
repair-primitive limits (shared cipher symbol, globally-coupled homophone).

## 1. Background: how a branch becomes declarable

`agent/loop_shared.py:43-97` — `_decoded_text_for_panel` prefers
`_metadata_decoded_text`, which returns (in order): the null-mask render when
`branch.key` AND a `null_mask_finalist`/`null_mask_selected` block exist
(lines 46-77), else `metadata["decoded_text"]` (line 78+). Attestation,
verify episodes, and `meta_declare_solution` all hash this string. The
quagmire3 and composite installers (`src/investigation/experiments.py:1576`,
`:1681`) therefore write `metadata["decoded_text"]` and **pop the two
null-mask blocks** (inherited masks would shadow the new text). The default
installer (`_install_experiment_branch`, `experiments.py:1546-1573`) predates
this pattern: it only does `set_full_key(...)` from `result["key"]`.

The automated-solver experiment runner (`experiments.py:250-319`) returns
`key` (flat substitution map from the artifact), `final_decryption`, and
`steps`. Mode-specific routes return an **empty** flat key by design — e.g.
the periodic screen (`src/automated/runner.py:3552-3571`) whose step even
says "The artifact key is intentionally empty", and the transposition route.
That is exactly the case the default installer mishandles.

## 2. P1 — default installer: decoded-branch install for empty-key results

**File:** `src/investigation/experiments.py`, `_install_experiment_branch`
(lines 1546-1573).

**Required behavior.** After the existing snapshot-restore and unique-name
logic, branch on the result shape:

1. `key` non-empty → **byte-identical to today**: `set_full_key`,
   `_mirror_null_mask_metadata`, return. No change.
2. `key` empty AND `str(result.get("final_decryption") or "").strip()`
   non-empty → **decoded-branch install** (the quagmire3/composite pattern):
   - do NOT `set_full_key` (leave the inherited snapshot key as-is);
   - `branch.metadata.pop("null_mask_finalist", None)` and
     `...pop("null_mask_selected", None)` (shadow hazard, same comment as
     `experiments.py:1629-1633`);
   - locate the *winning step*: the last entry in `result["steps"]` whose
     `"solver"` equals `result["solver"]`, else the last step, else `{}`;
   - `branch.metadata.update` with:
     - `cipher_mode`: `"periodic_polyalphabetic"` if the winning step's
       `key_type` is `"PeriodicShiftKey"` or its name contains `"periodic"`
       or `"vigenere"`; `"transposition"` if its name/solver contains
       `"transposition"`; else `"experiment_decoded"`;
     - `mode_status: "active"`, `mode_confidence: "medium"`;
     - `mode_evidence`: `f"Installed by experiment_collect from automated_solver experiment {record['experiment_id']} (solver {result.get('solver')!r})."`;
     - `mode_counter_evidence`: `"Bounded automated search can overfit; verify readability before declaring."`;
     - `key_type`: winning step's `key_type` (when present);
     - `mode_key_state`: a dict copied from the winning step's
       `{"variant", "period", "key", "shifts"}` entries that are present —
       this is where the recovered periodic key (fs5's `IHBMEP`) becomes
       visible to `decode_show`/branch cards;
     - `decoded_text`: the stripped `final_decryption`;
     - `decoded_text_source: "experiment_collect:automated_solver"`;
     - `search_metadata`: `{"solver", "experiment_id", "step_name"}`.
   - `workspace.tag(name, "hypothesis")`.
3. `key` empty AND `final_decryption` empty/blank → **byte-identical to
   today** (installs the snapshot copy with the empty key set). A failed
   experiment stays a visibly failed install; do not invent an error path.

**Tests** (new, `tests/test_experiments_install.py` or the existing
experiments test module — follow the current test file layout):
- a fake `automated_solver` record with `key={}`,
  `final_decryption="THERETIRED..."`, and a periodic step
  (`key_type="PeriodicShiftKey"`, `key="IHBMEP"`, `period=6`,
  `variant="vigenere"`) installs a branch where
  `_decoded_text_for_panel(...) == final_decryption`, metadata carries
  `cipher_mode="periodic_polyalphabetic"`, `mode_key_state["key"]=="IHBMEP"`,
  and both null-mask blocks inherited from the snapshot are gone;
- a record with a non-empty `key` installs exactly as before (assert
  `set_full_key` effect and NO `decoded_text` in metadata);
- a record with empty key and empty decryption behaves as today (no
  `decoded_text`);
- content-hash reachability: `_candidate_content_hash` over the installed
  branch's panel text equals the hash of `final_decryption` (the gate binds
  the right string).

## 3. P2 — composite padding-trim as an additional ranked candidate

**Files:** `src/investigation/experiments.py` — composite runner
(`_composite_substitution_transposition_runner`, lines 440-510), composite
installer (lines 1681-1779), `candidate_rank` dispatch guard (lines
1884-1895), and the `experiment_collect` model-facing description around
line 1267.

**Runner.** After computing `decryption`, detect a padding tail: the maximal
trailing run of ONE identical character with `4 <= run_len <= 32` and
`run_len < len(decryption)` (whitespace-stripped text; composite output is
boundary-less). When detected, add to the returned dict:

```python
"top_candidates": [
    {"rank": 1, "plaintext": decryption, "padding": None},
    {"rank": 2, "plaintext": trimmed, "padding": {"char": c, "length": run_len}},
]
```

When not detected, emit `"top_candidates"` with the single rank-1 entry.
`final_decryption` stays the FULL decryption (backward compatible;
zero-risk-additional-candidate design per the workplan).

**Installer.** Accept `candidate_rank` (default 1). Rank 1 must remain
byte-identical to today's install. Rank N>1 installs the selected
candidate's plaintext as `decoded_text` and additionally records
`"padding_trimmed": candidate["padding"]` in the branch metadata and
mentions the trim in `mode_evidence` (e.g. "… rank 2, padding tail
'M'×4 trimmed"). Out-of-range rank → structured `{"error": ...}` exactly in
the quagmire3 style (lines 1594-1600). Records produced by the OLD runner
(no `top_candidates`) must keep installing exactly as today.

**Dispatch guard.** Replace the hardcoded
`record.get("type") != "quagmire3_shotgun"` (line 1888) with a check that
the record's result actually carries ranked finalists
(`result.get("top_candidates")` truthy) — `candidate_rank` on an
`automated_solver` record must still be rejected with the current message
shape.

**Collect surface.** Where `experiment_collect` builds its payload for the
composite type, include a compact `candidates` summary (rank, length,
padding note) so the lead/MCP agent can SEE that a trimmed rank-2 exists.
Also extend the type's collect-description text (line ~1267 region) to name
`candidate_rank` for `composite_substitution_transposition`.

**Tests:**
- runner-level: monkeypatch
  `automated.runner._run_composite_substitution_transposition` to return a
  decryption ending in `"MMMM"` → result has 2 candidates, rank-2 trimmed,
  padding block correct; a run of 3 (`"MMM"`) or a whole-string run yields 1
  candidate;
- installer: rank 1 byte-identical (decoded_text == full), rank 2 installs
  trimmed text + `padding_trimmed` metadata; rank 3 → error; old-shape
  record (no top_candidates) + explicit candidate_rank → rejected by
  dispatch as today; rank on automated_solver → rejected;
- gate-reachability of the rank-2 branch (panel text == trimmed).

**Post-land acceptance (not coder scope):** re-run the round-4 dogfood
through the MCP surface; the sealed answer is
`~/.config/decipher/dogfood_answers/round4_composite_answer.json`.
Expected: rank-2 install verifies positively and `meta_declare_solution`
is ACCEPTED (this is the workplan's acceptance criterion).

## 4. P3 — explicit fractionation label suppresses the composite auto-route

**File:** `src/automated/runner.py`, Slice-B content auto-route (lines
2927-2975).

**Required behavior.** Add a module-level helper:

```python
def _names_fractionation_family(cipher_system: str) -> bool:
    cs = str(cipher_system or "").lower()
    return any(tok in cs for tok in (
        "bifid", "trifid", "adfgvx", "adfgx", "polybius",
        "fractionation", "fractionated",
    ))
```

Gate the `order_layer_suspected` block (line 2957) on
`not _names_fractionation_family(cipher_system)`: an explicitly labeled
fractionation cipher must NOT route to
`composite_substitution_transposition` (the peel targets substitution+
transposition and honest-fails on fractionation, wasting the budget and —
as fs7 showed — confusing the agent). With the label present the routing
falls through to the existing default exactly as if the signal had not
fired. No change to the submit-time misroute guard in
`experiments.py:598-621` (there is no dedicated fractionation type to
redirect to; the automated screens may still legitimately run).

**Tests:** a routing-level test (same style as the existing Slice-B route
tests) asserting `cipher_system="bifid"` with a text that trips
`order_layer_suspected` does NOT return the composite route, while the same
text with `cipher_system=""` still does. Include one non-suppressed control
(`cipher_system="unknown"`).

## 5. P4 — structured "no base key" error on repair of decoded branches

**Files:** `src/mcp_server/repair.py` (`compile_hypotheses`, and the
transaction entry) and/or `src/investigation/host.py`
(`check_repair_preconditions`) — put the guard at the SHARED seam
(`check_repair_preconditions`) if that covers both the MCP tools and the v3
internal repair path without disturbing existing precondition tests; add a
local guard in `compile_hypotheses` regardless (it runs before any compile
work).

**Required behavior** *(amended per review finding #3 — the original
"empty `branch.key`" condition was bypassable, because decoded installers
keep the inherited snapshot key as-is)*. The guard fires when the branch's
panel text is served from `metadata["decoded_text"]`, mirroring
`_metadata_decoded_text`'s precedence exactly: if `branch.key` is non-empty
AND a `null_mask_finalist`/`null_mask_selected` block with a non-empty mask
exists, the mask+key render wins and repair proceeds (return None); else if
`metadata["decoded_text"]` is non-empty, return a structured error instead
of proceeding (regardless of any inherited key); else proceed:

```json
{"status": "error", "reason": "decoded_branch_no_base_key",
 "detail": "Branch '<name>' is a decoded-text install (<cipher_mode>) with no per-symbol base key; word repair operates on a base key and cannot polish this branch. Re-run the originating experiment with adjusted config, or rebuild the mapping with act_* tools. Known limitation - see docs/specs/declaration_hardening_spec.md P4."}
```

Keyed branches (including partial keys) are untouched. This is the
"log the limitation" deliverable from the workplan — a named, teachable
failure instead of a silent no-op compile.

**Docs:** add the limitation to `docs/mcp_onboarding.md` (repair section):
decoded-text branches (quagmire3 / composite / periodic experiment installs)
have no base key; repair tools will return `decoded_branch_no_base_key`.

**Tests:** MCP-surface test: `repair_hypotheses_test` against a branch with
`key={}` + `metadata["decoded_text"]` set → the structured error;
against a partial-key branch → unchanged behavior. If the guard lands in
`check_repair_preconditions`, add the equivalent host-level test.

## 6. Constraints

- **No firewall changes.** Nothing here may read ground truth; all new
  metadata derives from solver results already in the experiment record.
- **Byte-compatibility:** P1 case 1/3 and P2 rank-1 installs must be
  byte-identical to current behavior; existing tests must pass unmodified
  (except tests that assert the old candidate_rank guard message for the
  composite type, if any — update those to the new capability-based check).
- Keep `TOOLS.md` untouched (no agent-tool signatures change); the MCP
  `experiment_collect` description string change is server-side text.
- Suite baseline: **1844 passed / 2 skipped** on main at `120f2eb` — the
  landing bar is that plus the new tests.

## 7. Review adjudication (2026-07-19, Fable review: LAND WITH FIXES)

- **Finding 1 (fixed):** stale "zero behavior change" docstring on
  `check_repair_preconditions` — amended to name the P4 gate.
- **Finding 2 (fixed):** old-shape-composite dispatch rejection now pinned by
  its own test (`test_p2_candidate_rank_dispatch_rejects_old_shape_composite`).
- **Finding 3 (fixed, spec amended):** the P4 guard condition was strengthened
  from "key empty" to "decoded_text is the active panel source" (see §5);
  an inherited snapshot key no longer exempts a decoded install, and null-mask
  branches (mask+key render) remain repairable. Tested both ways.
- **Finding 4 (accepted):** failed-but-completed composite results now carry a
  rank-1 `top_candidates` entry, so an explicit `candidate_rank` on such a
  record reaches the installer's structured error instead of the old dispatch
  usage error; a zero-candidate quagmire3 record with a rank arg gets the
  generic capability error. Message drift only — both paths remain structured
  errors; sanctioned by the capability-based guard direction.
- **Finding 5 (accepted):** the collect-packet finalist summary gains
  `length`/`padding` keys for ALL ranked-finalist types (quagmire3 included,
  `padding` null there). Additive payload keys; one shared summary path is
  preferred over a composite-only parallel structure.
- **Finding 6 (fixed):** partial-key negative control added
  (`test_decoded_branch_guard_inherited_key_and_mask_precedence` case c).

## 8. Post-land acceptance — PASSED (2026-07-19)

Landed as `39b889a`; acceptance run same day through the MCP surface
(investigation `4a382b3cd40c`, post-restart server on the new code):
`composite_substitution_transposition` experiment `b19a8a57ec02` solved
round-4 in 18.2s (peeled 7-column columnar, keyword slot BAFDCEG; dict_rate
0.93); the collect packet surfaced the P2 candidates summary (rank 1 len 287;
rank 2 len 283, padding M×4); rank-2 install `composite_trimmed` → positive
independent attestation (coherence 9, reader_accepts_as_solution=true,
language confidence 0.99, $0.011) → `meta_declare_solution` **ACCEPTED**,
terminal status `solved`. Declared text byte-identical to the sealed answer's
`plaintext_letters` minus the MMMM tail
(`~/.config/decipher/dogfood_answers/round4_composite_answer.json`). The
same cipher failed DECL-1 on the padded tail pre-fix (memory
`composite-repair-padding-todo` B) — the workplan acceptance criterion
("round-4 declares clean") is met.

## 9. Slicing / landing

One phase, one coder task (P1-P4 are small and share files), one commit:
"Declaration hardening: decoded-branch installs, composite padding-trim,
fractionation route guard, repair limitation error (P1-P4)". Fable review of
the full diff vs this spec before commit, per CLAUDE.md.
