# Spec: `quagmire3_shotgun` experiment type + installable Quagmire results

Status: ready to implement
Author: Fable (main session), 2026-07-18
Motivation: dogfood round 6, investigation `bbd8eabb899b` — see
`docs/evidence/mcp_dogfood_results.md` §"Round 6". Two confirmed surface gaps:

1. **Silent misroute:** `experiment_submit(type="automated_solver",
   config={"cipher_system": "quagmire3"})` is accepted, but
   `automated/runner.py::_select_solver_path` matches the `"quag"` token and
   routes to the generic `periodic_polyalphabetic_screen`, which cannot solve a
   keyed tableau. No error, no route to the Rust shotgun engine
   (`analysis/polyalphabetic_fast.py::search_quagmire3_shotgun_fast`) that
   solves this family.
2. **No install path for a Quagmire solution:** `experiment_collect(install=
   true)` installs results via `workspace.set_full_key` (a `dict[int,int]`
   substitution key). Quagmire III solutions are *mode-specific decoded
   branches* (metadata `decoded_text` + `cipher_mode="quagmire3"` + key state),
   exactly as `tools_v2._tool_search_quagmire3_keyword_alphabet` installs them.
   With no metadata install path, an exact Quagmire solve cannot reach the
   branch-bound verify→declare gate (round 6 ended `unsolved` on a solved
   cipher).

One change in `src/investigation/experiments.py` closes both gaps for BOTH
surfaces: the v3 lead loop (`host.py`) and the MCP server
(`src/mcp_server/tools.py`) import the same `EXPERIMENT_SUBMIT_TOOL` /
`EXPERIMENT_COLLECT_TOOL` definitions and dispatch through the same
`dispatch_experiment_submit` / `dispatch_experiment_collect`.

**Gate semantics do NOT change.** The verify→declare gate (M5.3 Slice 6,
`state.attestation_is_positive`, content-hash binding via
`_decoded_text_for_panel`) is untouched; this spec only makes an existing
branch *shape* (mode-specific decoded branch, already first-class in v2 and in
the attestation renderer) reachable from the experiment queue. No provenance-
ledger entry changes form; no update to
`docs/mcp_policy_provenance_ledger.md` is required. The ground-truth firewall
is preserved: the new runner receives only (cipher, snapshot, config), and
`language` remains host-derived.

All file references below are repo-relative; line numbers are at HEAD
`ecb3b74`.

---

## Part 1 — new experiment type `quagmire3_shotgun`

In `src/investigation/experiments.py`, register a second entry in
`EXPERIMENT_TYPES` (currently line ~232, only `automated_solver`).

### 1.1 Config schema (`_QUAGMIRE3_SHOTGUN_SCHEMA`)

Same local validator dialect as `_AUTOMATED_SOLVER_SCHEMA`
(type/properties/enum/items; `additionalProperties: false` is documentation +
provider contract; the host-side belt is the whitelist in
`validate_experiment_config`, which already derives `allowed` from
`schema["properties"]` ∪ `_UNIVERSAL_CONFIG_KEYS` — no change needed there).

```python
_QUAGMIRE3_SHOTGUN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "keyword_lengths":   {"type": "array", "items": {"type": "integer"}},
        "cycleword_lengths": {"type": "array", "items": {"type": "integer"}},
        "hillclimbs":        {"type": "integer"},
        "restarts":          {"type": "integer"},
        "model_variant":     {"type": ["string", "null"]},
    },
}
```

Defaults (`_QUAGMIRE3_SHOTGUN_DEFAULTS`) — the winning agent-budget values from
round 6 ("recovers it 100% in all top-5 finalists, ~109 s"):

```python
{
    "keyword_lengths": [7],
    "cycleword_lengths": [8],
    "hillclimbs": 5000,
    "restarts": 250,
    "model_variant": None,
}
```

Bounds are enforced by **clamping in the runner** (matching the v2 tool's
style at `tools_v2.py:7059-7061`), not by schema (the local dialect has no
min/max). Clamps:

- `hillclimbs`: `max(1, min(int(v), 50_000))`
- `restarts`: `max(1, min(int(v), 5_000))`
- `keyword_lengths` / `cycleword_lengths`: keep entries `2 <= n <= 20`,
  truncate each list to 8 entries; an emptied list falls back to its default.
  (A non-integer entry is a schema type error → structured validation error at
  submit, not a clamp.)

`model_variant` stays universal (already folded in by
`model_facing_config_schema` and validated at submit by `_resolve_variant`).
It does not affect quadgram scoring (the engine loads
`ngram.NGRAM_CACHE.get(language, 4)` directly) — the field-doc should say
"accepted for queue uniformity; does not affect the quadgram engine".

Field docs (`_FIELD_DOCS["quagmire3_shotgun"]`), concise per-field prose in the
style of `_AUTOMATED_SOLVER_FIELD_DOCS`:

- `keyword_lengths` — "Candidate tableau-keyword lengths to sweep (default
  [7]). When the keyword length is unknown, sweep a small range, e.g.
  [5,6,7,8]. Entries clamped to 2–20, max 8 entries."
- `cycleword_lengths` — "Candidate cycleword lengths = periods (default [8]).
  Take this from period evidence (periodic IC / Kasiski) rather than sweeping
  blindly. Entries clamped to 2–20, max 8 entries."
- `hillclimbs` — "Hillclimb steps per restart (default 5000, the budget that
  solves the reference Quagmire III cases; clamped to 1–50000)."
- `restarts` — "Independent shotgun restarts per (keyword_length,
  cycleword_length) pair (default 250; clamped to 1–5000). nominal proposals =
  len(keyword_lengths)·len(cycleword_lengths)·restarts·hillclimbs."

### 1.2 Runner `_quagmire3_shotgun_runner(cipher, snapshot, config)`

Pure over inputs, mirroring `_automated_solver_runner` (lines 173-228):

1. Deep-copy the snapshot, `_restore_branch_into` a throwaway `Workspace`,
   derive `effective = ws.effective_cipher_text(branch_name)`.
2. Lazy-import; if the Rust kernel is unavailable, fail LOUDLY with guidance:

   ```python
   try:
       from analysis.polyalphabetic_fast import search_quagmire3_shotgun_fast
       result = search_quagmire3_shotgun_fast(...)
   except Exception as exc:
       raise RuntimeError(
           f"Rust quagmire3 shotgun engine unavailable or failed: "
           f"{type(exc).__name__}: {exc}. Build it with "
           f"`scripts/build_rust_fast.sh` and verify with `decipher doctor`."
       ) from exc
   ```

   The queue machinery already converts a runner exception into a `failed`
   record with `_format_exception` (which keeps only the final exception line —
   hence packing the guidance INTO the message) and `resubmit` works on failed
   records. Do not add a Python-screen fallback: per the v2 tool's
   `engine_equivalence` note, the Python screen is not an equivalent path and
   silently substituting it would repeat the round-6 silent-misroute mistake.
3. Call with clamped config:

   ```python
   search_quagmire3_shotgun_fast(
       effective,
       language=str(config.get("language") or "en"),
       keyword_lengths=<clamped>,
       cycleword_lengths=<clamped>,
       hillclimbs=<clamped>,
       restarts=<clamped>,
       seed=1,
       top_n=5,
       threads=0,   # inner threading governed by the engine/env, same as v2 tool default
   )
   ```

   `slip_probability` / `backtrack_probability` / `initial_keywords` stay at
   engine defaults — deliberately NOT exposed (keep the experiment surface to
   the four keys named by the motivating fix; blind context-seeding via
   initial_keywords is a v2-tool affordance, not an experiment one).
4. Return an artifact-bound dict shaped so the existing collect/summary
   machinery works unmodified:

   ```python
   {
       "status": result.get("status") or "completed",   # engine returns "completed"
       "solver": "quagmire3_shotgun_rust",
       "error_message": result.get("error"),            # None on success
       "elapsed_seconds": <measured, round 3>,
       "key": {},                                       # no substitution key for this family
       "final_decryption": <best candidate plaintext or "">,
       "top_candidates": [ ... up to 5 entries ... ],
       "steps": [ <one synthetic step dict, below> ],
       "nominal_proposals": <len(kw)*len(cw)*restarts*hillclimbs>,
   }
   ```

   Each `top_candidates[i]` keeps the engine candidate's fields used by the v2
   installer, copied defensively: `rank` (1-based), `score`,
   `selection_score`, `period`, `plaintext` (FULL text — the installer needs
   it), `preview`, and `metadata` (`alphabet_keyword`, `cycleword`,
   `cycleword_shifts`, `plaintext_alphabet`, `ciphertext_alphabet`,
   `quagmire_type`, `start_type` — pass through whatever subset the engine
   provides, plus candidate-level `key`/`shifts` fallbacks exactly as the v2
   tool reads them at `tools_v2.py:7231-7259`). Drop everything else
   (e.g. bulky engine internals) to keep the record/state small.

   The synthetic step (so `_route_and_primary_steps` renders a sensible
   `route_step`/`primary_step` in the collect packet):

   ```python
   {
       "name": "search_quagmire3_shotgun",
       "status": result.get("status") or "completed",
       "engine": "rust_shotgun",
       "keyword_lengths": ..., "cycleword_lengths": ...,
       "hillclimbs": ..., "restarts": ...,
       "nominal_proposals": ...,
       "candidates": len(top_candidates),
       "best_score": <top candidate score or None>,
   }
   ```

   `final_decryption` feeds `_completed_summary`'s 160-char preview — that is
   what makes the queue status line legible.

### 1.3 Registry entry

```python
EXPERIMENT_TYPES["quagmire3_shotgun"] = {
    "config_schema": _QUAGMIRE3_SHOTGUN_SCHEMA,
    "config_defaults": dict(_QUAGMIRE3_SHOTGUN_DEFAULTS),
    "runner": _quagmire3_shotgun_runner,
    "installer": _install_quagmire3_branch,          # NEW key, see Part 2
    "description": (
        "Blake-style Quagmire III / keyed-tableau shotgun search on the Rust "
        "engine (the same one behind search_quagmire3_keyword_alphabet). Use "
        "for periodic ciphers where a keyed tableau is suspected (period known "
        "from Kasiski/periodic IC, flat-ish frequency within phases). Results "
        "install as mode-specific decoded branches (no substitution key)."
    ),
}
```

Existing generic machinery that must keep working WITHOUT modification
(verify by tests, not by code changes): `apply_config_defaults`,
`dedup_key` (config-aware; distinct budgets dedup separately),
`validate_experiment_config`'s whitelist derivation, `_resolve_variant`,
resubmit, orphaning, the W/S/I arbiter, and `available_types` listings
(`sorted(EXPERIMENT_TYPES)` picks up the new type automatically).

## Part 2 — type-aware install in `experiment_collect`

### 2.1 Installer seam

Add an optional `installer` key to `EXPERIMENT_TYPES` entries. In
`dispatch_experiment_collect` (line ~1283), replace the direct call:

```python
installer = EXPERIMENT_TYPES.get(record["type"], {}).get("installer") \
    or _install_experiment_branch
installed_name = installer(workspace, record, args.get("as_name"), turn,
                           candidate_rank=<see 2.3>)
```

`_install_experiment_branch` keeps its current key-based behavior for
`automated_solver` (add a `candidate_rank` kwarg it accepts and ignores, or
route the kwarg only to installers that take it — implementer's choice, but a
`candidate_rank` supplied for an automated_solver collect must produce a
structured error, not be silently ignored: return
`{"error": "candidate_rank is only supported for experiment types with ranked
finalists (quagmire3_shotgun)"}`).

### 2.2 `_install_quagmire3_branch(workspace, record, as_name, turn, candidate_rank=1)`

Mirrors the v2 install block (`tools_v2.py:7229-7298`) but through the
experiment-snapshot path (source branch never mutated, same as
`_install_experiment_branch`):

1. `result = record["result"]`; `candidates = result.get("top_candidates") or []`.
   If empty → structured error `{"error": "experiment produced no quagmire3
   candidates to install"}` (dispatch returns it; record still marked
   collected — match the existing failed/orphaned collect behavior of being
   adjudicated once).
2. Select `candidates[candidate_rank - 1]`; out-of-range → structured error
   naming the valid range.
3. Fresh name: same scheme as `_install_experiment_branch`
   (`as_name` or `exp_<id6>_<source_branch>`, `_2`/`_3` suffix on collision).
4. Deep-copy `record["snapshot"]`, `_restore_branch_into` under the fresh
   name, `created_iteration = turn`. Do NOT `set_full_key` — the inherited
   snapshot key is left as-is; `_decoded_text_for_panel`
   (`agent/loop_shared.py:97`) prefers `metadata["decoded_text"]`, which is
   what attestation/verify/declare hash (this is the exact property that makes
   the gate reachable).
5. Update the new branch's metadata with the v2-parity block:

   ```python
   {
       "cipher_mode": "quagmire3",
       "mode_status": "active",
       "mode_confidence": "medium",
       "mode_evidence": f"Installed by experiment_collect from quagmire3_shotgun "
                        f"experiment {record['experiment_id']} (alphabet keyword "
                        f"{alphabet_keyword!r}, cycleword {cycleword!r}, rank "
                        f"{candidate_rank}).",
       "mode_counter_evidence": "Bounded shotgun search can overfit; verify "
                                "readability before declaring.",
       "key_type": "QuagmireKey",
       "quagmire_type": metadata.get("quagmire_type", "quag3"),
       "alphabet_keyword": ..., "cycleword": ...,
       "cycleword_shifts": metadata.get("cycleword_shifts", candidate.get("shifts")),
       "plaintext_alphabet": ..., "ciphertext_alphabet": ...,
       "quagmire_score": candidate.get("score"),
       "quagmire_selection_score": candidate.get("selection_score"),
       "decoded_text": candidate.get("plaintext", ""),
       "decoded_text_source": "experiment_collect:quagmire3_shotgun",
       "search_metadata": {
           "solver": "quagmire3_shotgun_rust", "engine": "rust_shotgun",
           "experiment_id": record["experiment_id"],
           "candidate_rank": candidate_rank,
           "hillclimbs": ..., "restarts": ...,
           "keyword_lengths": ..., "cycleword_lengths": ...,
           "nominal_proposals": result.get("nominal_proposals"),
       },
   }
   ```

   (`alphabet_keyword`/`cycleword` fall back to `"unknown"` /
   `candidate.get("key")` exactly like the v2 tool.)
6. Tags: `hypothesis`, `mode:quagmire3`, `mode:keyed_tableau_polyalphabetic`.
7. Return the installed name. The surrounding dispatch code (hypothesis-board
   card mirroring, `installed_as` stamping, packet assembly) is unchanged and
   applies as-is. `_maybe_create_null_mask_session` finds no
   `search_null_masks` step and no-ops — fine.

### 2.3 `candidate_rank` in the collect tool

Add to `EXPERIMENT_COLLECT_TOOL["input_schema"]["properties"]`:

```python
"candidate_rank": {"type": "integer", "description":
    "1-based finalist to install for ranked-result experiments "
    "(quagmire3_shotgun). Default 1 = best."},
```

Dispatch reads `max(1, int(args.get("candidate_rank") or 1))`, passes it to
the installer only when `install=true`. Mention it in the tool description's
install sentence. The collect packet for a completed `quagmire3_shotgun`
should additionally include `"candidates": [{rank, score, selection_score,
preview} ...]` (NOT full plaintexts — `decoded_preview` already carries the
best) so a lead can choose a rank without a second tool. Implement via a
small type-conditional block in packet assembly keyed off
`record["type"] == "quagmire3_shotgun"` (or generically: include when the
result carries `top_candidates`; generic preferred).

## Part 3 — misroute guard on `automated_solver`

In `validate_experiment_config` (line ~261): after the whitelist loop and
schema check, add a type-specific guard for `automated_solver`:

```python
if exp_type == "automated_solver":
    cs = str(config.get("cipher_system") or "").lower()
    if "quag" in cs:
        errors.append(
            f"cipher_system {config.get('cipher_system')!r} names the "
            "Quagmire/keyed-tableau family, which the automated_solver stack "
            "cannot solve (it would silently misroute to the generic periodic "
            "screen). Submit type='quagmire3_shotgun' instead."
        )
```

And in `corrected_config_example`: drop a `cipher_system` value containing
`"quag"` (case-insensitive) before assembling the example, so the returned
example is genuinely valid (the existing final firewall check would otherwise
collapse it to `{}`; dropping the key yields a more useful example). The error
response's existing `note` field in `dispatch_experiment_submit` need not
change; the appended error text carries the redirect.

Scope decision (recorded): the guard covers only the Quagmire token family —
the one hint class that is *accepted but unsolvable* by the automated stack.
Other free-form hints keep their current routing (they all reach a solver that
is at least plausibly applicable); widening hint validation to a closed enum is
out of scope and would risk breaking existing callers.
`automated/runner.py::_select_solver_path` is deliberately NOT modified: it
serves the CLI/benchmark path whose result contract (substitution `key` dict)
a Quagmire solution cannot satisfy; the experiment surface is the right layer
for the redirect.

## Part 4 — submit tool schema for two types

`EXPERIMENT_SUBMIT_TOOL` (line ~825):

- `description`: extend to name both types, one sentence each:
  keep the existing `automated_solver` sentence; add "Type `quagmire3_shotgun`
  runs the Rust keyed-tableau/Quagmire III shotgun search (use when period
  evidence suggests a keyed tableau; results install as decoded branches via
  experiment_collect)."
- `type` property description: "Supported: 'automated_solver' (the no-LLM
  solver stack) and 'quagmire3_shotgun' (Rust Quagmire III keyed-tableau
  search). Each type's `config` is validated against its own schema."
- `config`: replace the single hardcoded
  `model_facing_config_schema("automated_solver")` with a union:

  ```python
  "config": {
      "anyOf": [
          {"title": "automated_solver config",
           **model_facing_config_schema("automated_solver")},
          {"title": "quagmire3_shotgun config",
           **model_facing_config_schema("quagmire3_shotgun")},
      ],
      "description": "Per-type config; see the branch matching your `type`.",
  },
  ```

  Host-side per-type validation remains the authoritative belt (unchanged).
  If any live provider rejects `anyOf` at a property position, fallback is a
  permissive `{"type": "object"}` with both contracts described in prose —
  but try `anyOf` first (OpenAI + Anthropic + MCP clients all accept it).

The MCP server surface (`src/mcp_server/tools.py:254-269`) imports these
definitions and wraps them in `_mutate_envelope` — no change there. Same for
`host.py` dispatch (`host.py:634-656`) — no change.

## Part 5 — tests (extend `tests/test_experiments.py`; one MCP-surface test in `tests/test_mcp_tools.py`)

Use the existing harness patterns (`_make_state`, `_arbiter_env_guard`,
synchronous queues, `monkeypatch`). Monkeypatch
`analysis.polyalphabetic_fast.search_quagmire3_shotgun_fast` (patch it where
the runner imports it) with a fake returning a two-candidate engine-shaped
result — no Rust dependency, no wall-clock cost.

1. **Registry/schema surface:** `quagmire3_shotgun` in `EXPERIMENT_TYPES`;
   `model_facing_config_schema("quagmire3_shotgun")` exposes exactly
   {keyword_lengths, cycleword_lengths, hillclimbs, restarts, model_variant}
   with defaults 5000/250/[7]/[8] and descriptions; submit-tool `config` schema
   is the anyOf union carrying both titles.
2. **Misroute guard:** automated_solver + `cipher_system="quagmire3"` (and
   `"Quagmire III"`) → structured error mentioning `quagmire3_shotgun`;
   `corrected_example` validates cleanly and lacks the quag hint; a
   non-quag hint (e.g. `"vigenere"`) still validates.
3. **Runner shape + purity:** patched engine; runner returns the Part-1.2
   shape (empty `key`, `final_decryption` = best plaintext, synthetic step,
   nominal_proposals); input snapshot byte-identical after run (mirror
   `test_automated_solver_runner_does_not_mutate_input_snapshot`).
4. **Clamps:** hillclimbs=10**9, restarts=0, keyword_lengths=[1,25,7],
   cycleword_lengths=[] → engine called with clamped/filtered/defaulted
   values (assert via the fake's captured kwargs).
5. **End-to-end install (sync queue):** submit → collect(install=true) →
   branch exists with `cipher_mode="quagmire3"`, `decoded_text` = best
   plaintext, tags present, `installed_as` stamped, packet carries
   `candidates` ranks; `_decoded_text_for_panel(workspace, name)` equals the
   candidate plaintext (this is the gate-reachability property).
6. **candidate_rank:** rank 2 installs candidate 2; rank 99 → structured
   error; `candidate_rank` on an automated_solver collect → structured error.
7. **Engine-unavailable:** patch the import target to raise → record
   `failed`, error mentions `build_rust_fast.sh`; `resubmit` yields a fresh
   pending record.
8. **No candidates:** engine returns empty `top_candidates` → collect
   install returns structured error, no branch created.
9. **Dedup:** same branch+config dedups (`deduplicated: True`); different
   `hillclimbs` does not.
10. **MCP surface (`tests/test_mcp_tools.py`):** `tools/list` (or the module's
    definition table) exposes `experiment_submit` whose description names
    `quagmire3_shotgun`, and the mutate envelope preserved the anyOf config.
    Follow the file's existing call/fixture pattern.

Full suite must stay green: `PYTHONPATH=src .venv/bin/python -m pytest
tests/ -q` (1773 passed / 2 skipped at HEAD, plus the new tests).

## Part 6 — docs

- `TOOLS.md` (v3 lead-tools section at the end): update the
  `experiment_submit` entry — add the `quagmire3_shotgun` type with its four
  config keys + defaults; update `experiment_collect` for `candidate_rank`
  and ranked-candidate packets.
- `CLAUDE.md`: no structural change required (experiment tools are described
  generically); do not touch counts.
- `docs/evidence/mcp_dogfood_results.md`: NOT part of this phase — the
  acceptance run appends its own entry afterwards.

## Review adjudication (Fable review, 2026-07-18 — approve-with-nits)

Amendments accepted into this spec and implemented:

1. **No-shadowing (installer):** the restored snapshot may inherit a
   `null_mask_finalist`/`null_mask_selected` block whose mask+key render takes
   precedence over `decoded_text` in `_metadata_decoded_text` — the installer
   must `pop` both blocks before setting the quagmire metadata, or the
   verify/declare hash binds the OLD null-mask decode.
2. **Empty-plaintext candidate:** installing a candidate with empty
   `plaintext` would leave `_decoded_text_for_panel` falling back to the
   inherited snapshot key (a stale decode). The installer returns a structured
   error instead; other ranks remain installable from the same record.

Recorded, deliberately NOT changed (follow-ups, not blockers):

3. The W/S/I arbiter's inner-worker cap does not reach the Rust shotgun
   engine (`threads=0` = all cores; the engine reads no env var), so a
   concurrent q3 experiment can oversubscribe CPU and
   `record["inner_workers"]` is nominal for this type. Backlog: thread the
   queue's `I` into the engine call.
4. `dedup_key` hashes the raw defaults-applied config while the runner
   clamps, so two configs clamping to the same effective run dedup
   separately (duplicate compute possible, never a false dedup).
5. `candidate_rank` on a failed/orphaned collect is ignored (the failed
   branch returns first); rank ≤ 0 clamps to 1 per the dispatch formula.
6. The non-ranked-type guard hardcodes `quagmire3_shotgun`; a future ranked
   type should introduce a registry capability flag instead.

## Out of scope (recorded deliberately)

- No change to `automated/runner.py` routing or the CLI/benchmark path.
- No Python-screen fallback engine for the experiment type.
- No `initial_keywords` / slip / backtrack / seed / top_n / threads exposure.
- No provenance-ledger changes (no gate semantics touched).
- No composite substitution+transposition work (backlog §6.2).
