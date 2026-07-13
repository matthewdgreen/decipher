# Spec: Improvement Program Phase 1 — CandidatePacket + FinalistSessionStore

Parent plan: `docs/improvement_program_plan.md` (Phase 1).
Spec author: Fable (main session). Implementer: coding sub-agent.
Design context: packets are the interchange format for the v3 episode
architecture (`docs/specs/agent_v3_design.md`, amendment A1 requires the
session store to be ownable by `InvestigationState` later — in this phase
it is owned by the executor, but must have no hard executor dependency).

## Scope and constraints

- Two deliverables: (1) a `CandidatePacket` dataclass + adapters from the
  three existing finalist-row shapes; (2) a single `FinalistSessionStore`
  replacing the three copy-pasted session subsystems in
  `src/agent/tools_v2.py`.
- **Zero model-visible change.** The JSON returned by every review/rate/
  install tool must be byte-identical in structure: same keys, same id
  strings, same ordering semantics. Packets are stored server-side and in
  artifacts, never added to model-visible review rows (token cost).
- The regression surface is exact-field-name tests:
  `tests/test_cipher_transformers.py::test_agent_transform_search_returns_finalist_review_and_branches`
  (asserts `search_session_id == "transform_search_1"` among others),
  `tests/test_polyalphabetic_fast.py:1033` (pure-transposition flow),
  `tests/test_homophonic_anneal.py:595` (null-mask flow). These must pass
  unchanged. Run the full suite before starting (expect 792 passed,
  1 skipped) and after; add no failures.
- Do not commit. Targeted edits only. Line numbers below verified
  2026-07-13 at commit `145b9fa`; locate by quoted code if drifted.

## Part A — `src/analysis/candidate_packet.py` (new)

```python
@dataclass
class CandidatePacket:
    packet_version: int = 1
    candidate_id: str            # stable per-source id (see adapters)
    kind: str                    # "null_mask" | "transform" | "pure_transposition"
    source: dict                 # generator provenance: solver, profile, config knobs
    rank: int | None             # 1-based rank in its menu at creation time
    text: str | None             # decoded text or best preview available
    preview: str | None          # short preview if distinct from text
    solver_scores: dict          # kind-native scalar scores (verbatim keys)
    validation: dict | None      # analysis.finalist_validation block, verbatim
    language_features: dict | None  # LANGUAGE_QUALITY_FEATURES-keyed dict when available
    provenance: dict             # mask / pipeline / key / params — enough to reproduce
    rating: dict | None          # agent_readability_judgment when present
    extras: dict                 # anything kind-specific not fitting above
```

With `to_dict()` / `from_dict()` (round-trip stable, unknown keys in
`from_dict` land in `extras`).

### Adapters (pure functions, same module)

Field mappings below use the verbatim source keys; missing keys → None /
empty dict, never KeyError.

1. `packet_from_null_mask_row(row, *, source_branch=None)` — input: a
   full finalist row from the bakeoff (`automated/runner.py` inner
   `finalist_row`, `:3801–3852`) or a compact row
   (`_compact_null_mask_row`, `:4980–5045`). `candidate_id`: use
   `row["candidate_id"]` when present, else `"mask:" + ",".join(mask)`.
   `solver_scores`: `{anneal_score, selection_score, validation_score_v2,
   confirmed_validation_score_v2, promoted_validation_score_v2,
   ensemble_score_v1, ensemble_vote_rate_v1}` (present keys only).
   `validation`: null-mask rows have no finalist_validation block — store
   None, and put `validation_components_v2` + `diagnostics` + `quality`
   into `extras`. `language_features`: map the compact row's
   `language_quality_*` fields when present
   (`language_quality_raw_score`, `language_quality_score`,
   `language_quality_rank_score`, `language_quality_model`) into
   `extras["language_quality"]`; `language_features` proper stays None
   unless a full feature dict is attached upstream. `text`:
   `row.get("decryption")`; `preview`: `row.get("preview")`.
   `provenance`: `{mask, mask_size, filtered_length, key}` (key may be
   large; include it — packets are artifact-side). `rating`:
   `row.get("agent_readability_judgment")`.
2. `packet_from_transform_row(row)` — input: a ranked row built at
   `tools_v2.py:11011–11021` (`candidate_index, candidate, pipeline,
   status, solver, anneal_score, elapsed_seconds, decoded_preview,
   validation, key`). `candidate_id`:
   `row["candidate"].get("candidate_id")` else
   `f"cand_{row['candidate_index']}"`. `solver_scores`:
   `{anneal_score}`. `validation`: verbatim. `provenance`: `{pipeline,
   key, family: candidate.get("family"), params: candidate.get("params")}`.
   `text`: None; `preview`: `decoded_preview`.
3. `packet_from_pure_transposition_row(row)` — input: a row from
   `analysis/pure_transposition.py:163–179` (`candidate.to_dict()` merged
   with `rank, rust_rank, score, selection_score, plaintext, preview` plus
   `validation`/`validated_selection_score` from the menu evaluation).
   `solver_scores`: `{score, selection_score, validated_selection_score,
   rust_rank}`. `provenance`: `{candidate_id, family, params, pipeline,
   inverse_mode, grid, provenance}` (verbatim from `to_dict`). `text`:
   `plaintext`; `preview`: `preview`.

### Tests (new `tests/test_candidate_packet.py`)

- Round-trip each adapter on a realistic fixture row (lift row literals
  from the three regression tests' mocked data).
- Unknown-key tolerance: adapter on a row with an extra field → lands in
  `extras`, `from_dict(to_dict(p)) == p`.
- No mutation: adapters must not mutate the input row (assert deep-equal
  before/after).

## Part B — `src/agent/finalist_sessions.py` (new) + refactor

### Current state (verified)

Three stores + counters on `WorkspaceToolExecutor.__init__`
(`tools_v2.py:2111–2116`); id schemes `f"transform_search_{n}"` (`:9951`),
`f"pure_transposition_{n}"` (`:9978`), `f"null_mask_{n}"` (`:9437`);
constructors `_new_transform_search_session` (`:9943`),
`_new_pure_transposition_session` (`:9971`), `_new_null_mask_session`
(`:9431`); getters `:9968`, `:9993`, `:9457`. Install helpers find their
session id by identity-scanning the store
(`:10202–10208`, `:10287`); the null-mask gate iterates
`reversed(self._null_mask_sessions.items())` (`:9637`);
`_null_mask_refinement_tried_for_branch` scans by `source_branch`
(`:3044`).

### Required shape

```python
class FinalistSessionStore:
    def new_session(self, kind: str, payload: dict, *, packets: list[CandidatePacket]) -> str
    def get(self, kind: str, session_id: str) -> dict | None
    def find_id(self, kind: str, payload: dict) -> str | None      # identity lookup
    def sessions(self, kind: str) -> Iterable[tuple[str, dict]]     # insertion order
    def last_sessions(self, kind: str) -> Iterable[tuple[str, dict]]  # reversed
```

- Id scheme preserved exactly: `f"{kind}_{n}"` with a per-kind counter
  starting at 1, where kind strings are `"transform_search"`,
  `"pure_transposition"`, `"null_mask"` (the null-mask id prefix is
  `null_mask_` even though the store key naming differs today — match the
  current emitted ids exactly).
- Session payload dicts keep their current per-kind keys verbatim (do NOT
  unify payload schemas in this phase). The store adds one key:
  `payload["packets"]` = list of packet dicts (via `to_dict()`).
- The store lives in its own module with no imports from `tools_v2` (the
  v3 design later moves ownership to `InvestigationState`; keep it
  dependency-free).
- Executor: `self._finalist_sessions = FinalistSessionStore()` replaces
  the three dicts + three counters. Keep the six `_new_*`/getter helpers
  as thin delegating wrappers (their call sites are numerous); replace
  identity reverse-lookups with `find_id`. The null-mask gate and
  `_null_mask_refinement_tried_for_branch` iterate via
  `sessions("null_mask")`/`last_sessions("null_mask")`.

### Packet attachment points

- Transform: in `_tool_search_transform_homophonic` where `ranked` is
  built (`:11011–11021`), map rows → `packet_from_transform_row`.
- Pure transposition: in `_new_pure_transposition_session`, map
  `result["top_candidates"]` → `packet_from_pure_transposition_row`.
- Null-mask: in `_new_null_mask_session`, map `result["top_finalists"]`
  → `packet_from_null_mask_row(row, source_branch=...)`.
- Ratings: the three rate tools already write
  `agent_readability_judgment` onto the underlying candidate rows; after
  rating, refresh the corresponding packet's `rating` field (match by
  rank/candidate_id).

### Artifact attachment (additive only)

- Runner side: in `automated/runner.py` where the `search_null_masks`
  step's `top_finalists` are finalized (`:4962`), attach
  `row["packet"] = packet_from_null_mask_row(row).to_dict()` to each
  top-finalist row. Same additive attachment for
  `screen_pure_transposition`'s `top_candidates` (`runner.py:2735` area /
  `analysis/pure_transposition.py` result rows) and the transform menu's
  `finalists` (`evaluate_finalist_menu` consumers — attach at the runner
  call site `runner.py:1329–1378`, not inside the shared skeleton).
- Import direction: `automated/runner.py` and `agent/tools_v2.py` import
  `analysis.candidate_packet`; never the reverse.
- Do NOT add packets to `ToolCall.result` payloads beyond what the
  existing review JSON already contains (they reach artifacts via session
  storage on install-metadata and the runner steps above).

### Tests

- Extend the three regression flows minimally (do not rewrite them): after
  driving each search tool, assert the session payload has `packets` with
  the expected count and that packet[0] round-trips.
- Store unit tests: id sequence stability per kind, `find_id` identity
  semantics, insertion/reversed iteration order.
- A test asserting `finalist_sessions.py` imports neither `tools_v2` nor
  `loop_v2` (dependency-freedom for v3).

## Acceptance

- Full suite green, no changes to the three regression tests' assertions.
- `grep -n '_transform_search_sessions\|_null_mask_sessions\|_pure_transposition_sessions' src/agent/tools_v2.py`
  shows only the thin wrappers/property aliases (or nothing), not three
  parallel implementations.
- One additional smoke: run
  `PYTHONPATH=src .venv/bin/python -m pytest tests/test_homophonic_anneal.py tests/test_cipher_transformers.py tests/test_polyalphabetic_fast.py -q`
  and report timing vs baseline.

## Review follow-ups (deferred)

From the Fable review (LAND WITH FIXES, applied 2026-07-13). Deferred to
Phase 2, where packet consumers are decided:

- **F3 packet size**: runner-side packets duplicate `text` (full
  decryption) and `provenance.key` that already exist as sibling keys on
  the same artifact row (~4.5KB/packet, ~2× growth on finalist sections,
  low single-digit MB per multi-page sweep). Phase 2 should either trim
  those fields from row-attached packets or gate them behind a flag —
  decide when the word-repair/multipage consumers exist.
- **F5 stale mirrors**: runner-row packets don't receive rating updates
  (only session packets do) and don't gain `branch` after install —
  acceptable now; revisit if artifact consumers start reading row packets.
- Adapters alias sub-objects by reference (F6) — fine for serialization;
  v3 must not mutate packets in place.

## Deliverables / final report

Files changed; suite before/after; any place the spec's field mappings did
not match reality (report, don't invent); confirmation that review JSON is
unchanged (diff one captured review payload before/after refactor for each
of the three kinds — e.g. by running the three regression tests' flows and
comparing the review dicts).
