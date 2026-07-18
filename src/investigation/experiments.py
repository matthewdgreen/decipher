"""Experiment queue: long-running solver compute the v3 lead delegates to (M4).

An *experiment* is a PURE FUNCTION over a deep-copied branch snapshot: it runs
the automated no-LLM solver stack (``run_automated``) in the background while
the lead keeps working, and its result is integrated back onto the lead thread
only when the lead explicitly collects it. Nothing an experiment does touches
the lead's live workspace until ``experiment_collect(install=true)``.

Key pieces (M4 spec):
- ``EXPERIMENT_TYPES`` registry (v1 type ``automated_solver``); the runner
  contract is ``(cipher, branch_snapshot, config) -> result dict`` (Part 1/A5).
- ``ExperimentQueue`` — daemon-thread execution with a lock-guarded single-
  writer result box and the W/S/I worker-budget arbiter (Part 2/A5).
- ``dispatch_experiment_submit`` / ``dispatch_experiment_collect`` — the two
  lead tools (Part 3), plus ``EXPERIMENT_TOOL_DEFINITIONS`` (assembled into the
  final lead tool list in ``loop_v3``, never appended into ``episodes.py``).

Honest daemon-thread exit note (F3): daemon-ness does NOT make process exit
free. ``run_automated``'s CPU fan-out opens its own ``ProcessPoolExecutor``
whose NON-daemon management threads are joined at interpreter exit regardless,
so a ``run_ended`` orphan can delay a non-interrupt exit until its solver
finishes. Ctrl-C exits promptly because the terminal SIGINTs the whole process
group (the seed children die) — NOT because of daemon threads. Orphaning
(Part 4) is a STATE transition only. Inner pools spawned from a worker thread
are safe under spawn (macOS default); flag fork flakiness on Linux.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
import traceback
import uuid
from typing import Any, Callable

from analysis import model_registry
from agent.tools_v2 import _strip_packet_keys
from investigation.board import CARD_MIRROR_KEYS
from investigation.episodes import validate_against_schema
from investigation.state import (
    InvestigationState,
    _restore_branch_into,
    _serialize_branch,
)
from workspace import Workspace


# ---------------------------------------------------------------------------
# Part 1 — experiment types + registry (C5 + A5)
# ---------------------------------------------------------------------------
# ``model_variant`` is a UNIVERSAL experiment key (validated at submit via the
# registry, stamped into the defaults-applied config so dedup is variant-aware);
# it is accepted for every type regardless of the type's own config schema.
_UNIVERSAL_CONFIG_KEYS = frozenset({"model_variant"})

# v1 ``automated_solver`` config schema. Local validator dialect
# (type/properties/enum, nullable via a type list) — the ``episodes`` validator.
# The ``transform_search`` enum EXCLUDES ``"promote"`` (F5): promote reads prior
# artifacts from disk (a purity break) and raises when its companion arg is
# absent.
#
# ``additionalProperties: false`` is declared here so the model-facing schema
# (Slice 5) can carry the provider-side unknown-key rejection verbatim. NOTE:
# the local ``validate_against_schema`` dialect does NOT interpret
# ``additionalProperties`` (it only understands type/enum/properties/required/
# items). Host-side unknown-key rejection therefore lives in the explicit
# whitelist in ``validate_experiment_config``; this key is documentation + the
# provider contract, not something the local validator enforces.
#
# ``language`` is deliberately NOT a property: it is host-derived
# (``config["language"] = state.language`` at dispatch) and the model must not
# supply it — an emitted ``language`` key is rejected as an unknown key.
_AUTOMATED_SOLVER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "homophonic_budget": {"type": "string", "enum": ["full", "screen"]},
        "homophonic_refinement": {
            "type": "string",
            "enum": [
                "none", "two_stage", "targeted_repair", "family_repair",
                "null_masks", "homophonic_nulls",
            ],
        },
        "homophonic_solver": {"type": "string", "enum": ["zenith_native", "legacy"]},
        "transform_search": {
            "type": "string",
            "enum": ["off", "auto", "screen", "wide", "rank", "full"],
        },
        "transform_search_profile": {"type": "string"},
        "transform_search_max_generated_candidates": {"type": ["integer", "null"]},
        "cipher_system": {"type": "string"},
        "model_variant": {"type": ["string", "null"]},
    },
}

# Concise, model-facing per-field help for ``automated_solver``. Surfaced in the
# ``experiment_submit`` tool schema (Slice 5) so the model sees the real contract
# instead of guessing keys/flags. Homophonic behavior is selected THROUGH these
# fields, never via free-form flags such as ``allow_homophones``; runtime
# controls such as ``max_runtime_seconds`` are not accepted.
_AUTOMATED_SOLVER_FIELD_DOCS: dict[str, str] = {
    "cipher_system": (
        "Cipher-family HINT that routes the automated solver, e.g. "
        "'simple_substitution', 'homophonic_substitution', 'vigenere', "
        "'transposition'. Free-form string; '' lets the router auto-detect."
    ),
    "homophonic_budget": (
        "Compute budget for the homophonic annealer: 'full' (default) or the "
        "cheaper 'screen' pass."
    ),
    "homophonic_refinement": (
        "Post-anneal refinement for homophonic ciphers. Select homophonic "
        "handling HERE (e.g. 'targeted_repair', 'null_masks'), not via free-form "
        "flags. 'none' disables refinement."
    ),
    "homophonic_solver": (
        "Homophonic solver engine: 'zenith_native' (Zenith-parity SA, default) "
        "or 'legacy'."
    ),
    "transform_search": (
        "Transform/transposition screen breadth over the branch. 'off' (default) "
        "disables it; 'auto'/'screen'/'wide'/'rank'/'full' widen the search."
    ),
    "transform_search_profile": (
        "Named profile for the transform search (e.g. 'broad'). Applies only when "
        "transform_search is enabled."
    ),
    "transform_search_max_generated_candidates": (
        "Optional integer cap on transform candidates; null = solver default."
    ),
    "model_variant": (
        "Language-model variant slug (null = host default). Universal across "
        "experiment types; validated against the run language at submit."
    ),
}

# v1 ``quagmire3_shotgun`` config schema. Same local validator dialect; bounds
# are enforced by CLAMPING in the runner (the dialect has no min/max), not here.
# ``language`` is host-derived and deliberately NOT a property.
_QUAGMIRE3_SHOTGUN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "keyword_lengths": {"type": "array", "items": {"type": "integer"}},
        "cycleword_lengths": {"type": "array", "items": {"type": "integer"}},
        "hillclimbs": {"type": "integer"},
        "restarts": {"type": "integer"},
        "model_variant": {"type": ["string", "null"]},
    },
}

# Concise, model-facing per-field help for ``quagmire3_shotgun``.
_QUAGMIRE3_SHOTGUN_FIELD_DOCS: dict[str, str] = {
    "keyword_lengths": (
        "Candidate tableau-keyword lengths to sweep (default [7]). When the "
        "keyword length is unknown, sweep a small range, e.g. [5,6,7,8]. Entries "
        "clamped to 2-20, max 8 entries."
    ),
    "cycleword_lengths": (
        "Candidate cycleword lengths = periods (default [8]). Take this from "
        "period evidence (periodic IC / Kasiski) rather than sweeping blindly. "
        "Entries clamped to 2-20, max 8 entries."
    ),
    "hillclimbs": (
        "Hillclimb steps per restart (default 5000, the budget that solves the "
        "reference Quagmire III cases; clamped to 1-50000)."
    ),
    "restarts": (
        "Independent shotgun restarts per (keyword_length, cycleword_length) "
        "pair (default 250; clamped to 1-5000). nominal proposals = "
        "len(keyword_lengths)*len(cycleword_lengths)*restarts*hillclimbs."
    ),
    "model_variant": (
        "Language-model variant slug; accepted for queue uniformity; does not "
        "affect the quadgram engine."
    ),
}

# Per-type field-doc registry (parallel to EXPERIMENT_TYPES). A type without an
# entry simply exposes its raw schema with no per-field prose.
_FIELD_DOCS: dict[str, dict[str, str]] = {
    "automated_solver": _AUTOMATED_SOLVER_FIELD_DOCS,
    "quagmire3_shotgun": _QUAGMIRE3_SHOTGUN_FIELD_DOCS,
}

_AUTOMATED_SOLVER_DEFAULTS: dict[str, Any] = {
    "homophonic_budget": "full",
    "homophonic_refinement": "none",
    "homophonic_solver": "zenith_native",
    "transform_search": "off",
    "transform_search_profile": "broad",
    "transform_search_max_generated_candidates": None,
    "cipher_system": "",
    "model_variant": None,
}

# Winning agent-budget values from round 6 (recovers the reference Quagmire III
# cases 100% in ~109 s).
_QUAGMIRE3_SHOTGUN_DEFAULTS: dict[str, Any] = {
    "keyword_lengths": [7],
    "cycleword_lengths": [8],
    "hillclimbs": 5000,
    "restarts": 250,
    "model_variant": None,
}


def _reconstruct_effective(cipher: Any, snapshot: dict[str, Any]) -> Any:
    """Reconstruct the DERIVED effective CipherText from a stored branch snapshot.

    Builds a one-branch throwaway ``Workspace`` over ``cipher`` and returns
    ``workspace.effective_cipher_text(branch)`` — exactly how the v2 tool and the
    ``automated_solver`` runner derive their input. Deep-copies the snapshot so
    the record's copy stays pristine.
    """
    ws = Workspace(cipher_text=cipher)
    snap = copy.deepcopy(snapshot)
    _restore_branch_into(ws, snap)
    return ws.effective_cipher_text(str(snap["name"]))


def _automated_solver_runner(
    cipher: Any, snapshot: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Runner for the ``automated_solver`` experiment type (Part 1).

    PURE over its inputs: reconstructs a throwaway one-branch workspace, derives
    the effective cipher, sets the registry's thread-local active variant for the
    duration (F1a per-thread-safe), and calls ``run_automated`` with
    ``ground_truth=None`` hardcoded (firewall). Returns only the artifact-bound
    fields listed in the spec.
    """
    from automated import runner as automated_runner  # lazy (Part 1)

    # F5: deep-copy the stored snapshot AGAIN before restore (restore shallow-
    # copies nested metadata); the record's snapshot must stay pristine.
    ws = Workspace(cipher_text=cipher)
    snap = copy.deepcopy(snapshot)
    _restore_branch_into(ws, snap)
    branch_name = str(snap["name"])
    effective = ws.effective_cipher_text(branch_name)

    variant = config.get("model_variant")
    # F1a: set the calling thread's active variant for the duration; run_automated
    # sets it again from its own param (belt-and-suspenders on the same thread).
    prev_variant = automated_runner.set_active_model_variant(variant)
    try:
        result = automated_runner.run_automated(
            cipher_text=effective,
            language=str(config.get("language") or "en"),
            cipher_id="experiment",
            ground_truth=None,  # firewall: hardcoded, never smuggled via config
            cipher_system=str(config.get("cipher_system") or ""),
            homophonic_budget=str(config.get("homophonic_budget") or "full"),
            homophonic_refinement=str(config.get("homophonic_refinement") or "none"),
            homophonic_solver=str(config.get("homophonic_solver") or "zenith_native"),
            transform_search=str(config.get("transform_search") or "off"),
            transform_search_profile=str(config.get("transform_search_profile") or "broad"),
            transform_search_max_generated_candidates=config.get(
                "transform_search_max_generated_candidates"
            ),
            model_variant=variant,
        )
    finally:
        automated_runner.set_active_model_variant(prev_variant)

    artifact = result.artifact or {}
    raw_key = artifact.get("key") or {}
    return {
        "status": result.status,
        "solver": result.solver,
        "error_message": result.error_message,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "key": {str(k): int(v) for k, v in raw_key.items()},
        "final_decryption": result.final_decryption,
        "steps": list(artifact.get("steps") or []),
    }


def _clamp_length_list(raw: Any, *, default: list[int]) -> list[int]:
    """Keep integer entries with 2 <= n <= 20, truncate to 8; an emptied list
    falls back to ``default`` (matching the v2 tool's clamp style). A non-integer
    entry is a schema type error at submit, not a clamp — so this coerces only."""
    values: list[int] = []
    for item in raw or []:
        n = int(item)
        if 2 <= n <= 20:
            values.append(n)
    values = values[:8]
    return values if values else list(default)


def _quagmire3_candidate_record(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    """Copy only the fields the v2 installer reads (tools_v2.py:7231-7259),
    defensively; drop bulky engine internals to keep the record/state small."""
    metadata = dict(candidate.get("metadata") or {})
    slim_meta = {
        "alphabet_keyword": metadata.get("alphabet_keyword"),
        "cycleword": metadata.get("cycleword"),
        "cycleword_shifts": metadata.get("cycleword_shifts"),
        "plaintext_alphabet": metadata.get("plaintext_alphabet"),
        "ciphertext_alphabet": metadata.get("ciphertext_alphabet"),
        "quagmire_type": metadata.get("quagmire_type"),
        "start_type": metadata.get("start_type"),
    }
    return {
        "rank": rank,
        "score": candidate.get("score"),
        "selection_score": candidate.get("selection_score"),
        "period": candidate.get("period"),
        "plaintext": candidate.get("plaintext", ""),
        "preview": candidate.get("preview"),
        "key": candidate.get("key"),
        "shifts": candidate.get("shifts"),
        "metadata": slim_meta,
    }


def _quagmire3_shotgun_runner(
    cipher: Any, snapshot: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Runner for the ``quagmire3_shotgun`` experiment type (Part 1).

    PURE over its inputs: reconstructs a throwaway one-branch workspace, derives
    the effective cipher, and calls the Rust keyed-tableau shotgun engine. Bounds
    are enforced by CLAMPING here (the local schema dialect has no min/max). No
    Python-screen fallback: the Python screen is not an equivalent path, so an
    unavailable/failed kernel fails LOUDLY (the round-6 silent-misroute lesson).
    """
    ws = Workspace(cipher_text=cipher)
    snap = copy.deepcopy(snapshot)
    _restore_branch_into(ws, snap)
    branch_name = str(snap["name"])
    effective = ws.effective_cipher_text(branch_name)

    hillclimbs = max(1, min(int(config.get("hillclimbs", 5000)), 50_000))
    restarts = max(1, min(int(config.get("restarts", 250)), 5_000))
    keyword_lengths = _clamp_length_list(config.get("keyword_lengths"), default=[7])
    cycleword_lengths = _clamp_length_list(config.get("cycleword_lengths"), default=[8])

    start = time.time()
    try:
        from analysis.polyalphabetic_fast import search_quagmire3_shotgun_fast

        result = search_quagmire3_shotgun_fast(
            effective,
            language=str(config.get("language") or "en"),
            keyword_lengths=keyword_lengths,
            cycleword_lengths=cycleword_lengths,
            hillclimbs=hillclimbs,
            restarts=restarts,
            seed=1,
            top_n=5,
            threads=0,  # inner threading governed by the engine/env (v2 default)
        )
    except Exception as exc:  # noqa: BLE001 - packed into a `failed` record
        raise RuntimeError(
            f"Rust quagmire3 shotgun engine unavailable or failed: "
            f"{type(exc).__name__}: {exc}. Build it with "
            f"`scripts/build_rust_fast.sh` and verify with `decipher doctor`."
        ) from exc
    elapsed = round(time.time() - start, 3)

    raw_candidates = list(result.get("top_candidates") or [])[:5]
    top_candidates = [
        _quagmire3_candidate_record(cand, idx)
        for idx, cand in enumerate(raw_candidates, 1)
    ]
    nominal = len(keyword_lengths) * len(cycleword_lengths) * restarts * hillclimbs
    status = result.get("status") or "completed"
    best_score = top_candidates[0].get("score") if top_candidates else None
    best_plaintext = top_candidates[0].get("plaintext") if top_candidates else ""
    step = {
        "name": "search_quagmire3_shotgun",
        "status": status,
        "engine": "rust_shotgun",
        "keyword_lengths": keyword_lengths,
        "cycleword_lengths": cycleword_lengths,
        "hillclimbs": hillclimbs,
        "restarts": restarts,
        "nominal_proposals": nominal,
        "candidates": len(top_candidates),
        "best_score": best_score,
    }
    return {
        "status": status,
        "solver": "quagmire3_shotgun_rust",
        "error_message": result.get("error"),  # None on success
        "elapsed_seconds": elapsed,
        "key": {},  # no substitution key for this family
        "final_decryption": best_plaintext or "",
        "top_candidates": top_candidates,
        "steps": [step],
        "nominal_proposals": nominal,
    }


# type -> {config_schema, config_defaults, runner, description, installer?}
EXPERIMENT_TYPES: dict[str, dict[str, Any]] = {
    "automated_solver": {
        "config_schema": _AUTOMATED_SOLVER_SCHEMA,
        "config_defaults": dict(_AUTOMATED_SOLVER_DEFAULTS),
        "runner": _automated_solver_runner,
        "description": (
            "Run the automated no-LLM solver stack (the same one the parity "
            "harness uses) on a branch in the background: homophonic anneal, "
            "transform screens, and null-mask bakeoffs via its options."
        ),
    },
    "quagmire3_shotgun": {
        "config_schema": _QUAGMIRE3_SHOTGUN_SCHEMA,
        "config_defaults": dict(_QUAGMIRE3_SHOTGUN_DEFAULTS),
        "runner": _quagmire3_shotgun_runner,
        "installer": None,  # bound to _install_quagmire3_branch below (defined later)
        "description": (
            "Blake-style Quagmire III / keyed-tableau shotgun search on the Rust "
            "engine (the same one behind search_quagmire3_keyword_alphabet). Use "
            "for periodic ciphers where a keyed tableau is suspected (period known "
            "from Kasiski/periodic IC, flat-ish frequency within phases). Results "
            "install as mode-specific decoded branches (no substitution key)."
        ),
    },
}


def register_experiment_type(name: str, entry: dict[str, Any]) -> None:
    """Register an experiment type (test seam; mirrors register_session_builder)."""
    EXPERIMENT_TYPES[name] = entry


def _config_defaults(exp_type: str) -> dict[str, Any]:
    entry = EXPERIMENT_TYPES.get(exp_type) or {}
    return dict(entry.get("config_defaults") or {})


def _config_schema(exp_type: str) -> dict[str, Any]:
    entry = EXPERIMENT_TYPES.get(exp_type) or {}
    return dict(entry.get("config_schema") or {"type": "object", "properties": {}})


def validate_experiment_config(exp_type: str, config: dict[str, Any]) -> list[str]:
    """Validate a user config: unknown-key whitelist PLUS schema (structured).

    Returns a list of human-readable errors (empty == valid). Never raises — a
    validation failure is a structured tool error, and this is a firewall
    surface (Part 1/Part 6).

    Slice 5: the explicit whitelist loop below IS the host-side unknown-key
    rejection. ``additionalProperties: false`` is advertised in the model-facing
    schema so the provider rejects unknown keys before dispatch, but the local
    ``validate_against_schema`` dialect cannot express/enforce it — so the belt
    that guarantees rejection at dispatch is this whitelist (host-derived
    ``language`` and the M5.2 smoke's ``target_language``/``allow_homophones``/
    ``max_runtime_seconds`` all fail here).
    """
    errors: list[str] = []
    if exp_type not in EXPERIMENT_TYPES:
        return [f"unknown experiment type: {exp_type!r}"]
    schema = _config_schema(exp_type)
    allowed = set((schema.get("properties") or {}).keys()) | set(_UNIVERSAL_CONFIG_KEYS)
    for key in config:
        if key not in allowed:
            errors.append(
                f"unknown config key {key!r}; allowed: {sorted(allowed)}"
            )
    errors.extend(validate_against_schema(config, schema, path="config"))
    # Misroute guard: the Quagmire/keyed-tableau family is ACCEPTED but unsolvable
    # by the automated stack (it would silently route to the generic periodic
    # screen). Redirect to the dedicated type. Scope: only this token family — the
    # one accepted-but-unsolvable hint class; other free-form hints keep routing.
    if exp_type == "automated_solver":
        cs = str(config.get("cipher_system") or "").lower()
        if "quag" in cs:
            errors.append(
                f"cipher_system {config.get('cipher_system')!r} names the "
                "Quagmire/keyed-tableau family, which the automated_solver stack "
                "cannot solve (it would silently misroute to the generic periodic "
                "screen). Submit type='quagmire3_shotgun' instead."
            )
    return errors


def _field_docs(exp_type: str) -> dict[str, str]:
    return dict(_FIELD_DOCS.get(exp_type) or {})


def model_facing_config_schema(exp_type: str) -> dict[str, Any]:
    """Build the provider-visible ``config`` schema for one experiment type.

    Starts from the registered type's actual validation schema and enriches each
    property with its ``default`` (from ``config_defaults``) and a concise
    ``description`` (from ``_FIELD_DOCS``). Folds in the universal keys
    (``model_variant``) even for types whose own schema omits them, and stamps
    ``additionalProperties: false`` so a strict provider rejects unknown keys
    before the call reaches dispatch (A9: validation failures are structured,
    never a crash). ``language`` is intentionally absent — it is host-derived.
    """
    schema = _config_schema(exp_type)
    defaults = _config_defaults(exp_type)
    docs = _field_docs(exp_type)
    props_in = schema.get("properties") or {}
    props: dict[str, Any] = {}
    for key, sub in props_in.items():
        entry = dict(sub)
        if key in defaults:
            entry["default"] = defaults[key]
        if docs.get(key):
            entry["description"] = docs[key]
        props[key] = entry
    # Fold universal keys that the type's own schema did not already list.
    for key in sorted(_UNIVERSAL_CONFIG_KEYS):
        if key not in props:
            entry: dict[str, Any] = {"type": ["string", "null"]}
            if key in defaults:
                entry["default"] = defaults[key]
            if docs.get(key):
                entry["description"] = docs[key]
            props[key] = entry
    return {
        "type": "object",
        "additionalProperties": False,
        "description": (
            "Per-type solver configuration (all keys optional; defaults shown). "
            "`language` is HOST-DERIVED from the run and must NOT be supplied. "
            "`cipher_system` is a family hint. Select homophonic behavior through "
            "the homophonic_* fields, not free-form flags. Unknown keys and "
            "runtime controls (e.g. max_runtime_seconds) are rejected."
        ),
        "properties": props,
    }


def corrected_config_example(
    exp_type: str, user_config: dict[str, Any] | None
) -> dict[str, Any]:
    """Return a VALID, family-consistent config the model can copy after a
    validation error.

    Keeps the user's supported keys whose values already validate, drops unknown
    keys (host-derived ``language``, guessed ``target_language`` /
    ``allow_homophones`` / ``max_runtime_seconds``) and drops values that fail
    the enum/type check (so the type default applies). When the rejected input
    signals homophonic intent, seeds the correct homophonic_* fields so the
    example demonstrates the supported mechanism. Firewall: the returned example
    is guaranteed valid (falls back to ``{}`` — all defaults — if anything is
    off), never raises.
    """
    if exp_type not in EXPERIMENT_TYPES:
        return {}
    schema = _config_schema(exp_type)
    props = schema.get("properties") or {}
    allowed = set(props.keys()) | set(_UNIVERSAL_CONFIG_KEYS)
    example: dict[str, Any] = {}
    for key, value in (user_config or {}).items():
        if key not in allowed:
            continue  # drop unknown/host-only key
        sub = props.get(key)
        if sub is None or not validate_against_schema(value, sub):
            example[key] = value  # supported key with an already-valid value
        # else: invalid enum/type value dropped -> the type default applies
    # A ``cipher_system`` naming the Quagmire family passes the string schema but
    # fails the misroute guard; drop it so the example stays genuinely valid AND
    # useful (the guard would otherwise collapse the whole example to {}).
    if "quag" in str(example.get("cipher_system") or "").lower():
        example.pop("cipher_system", None)
    if _looks_homophonic(user_config):
        example.setdefault("cipher_system", "homophonic_substitution")
        example.setdefault("homophonic_solver", "zenith_native")
    if validate_experiment_config(exp_type, example):
        return {}  # firewall: never return an example that would itself fail
    return example


def _looks_homophonic(user_config: dict[str, Any] | None) -> bool:
    """Heuristic: did the (rejected) input signal homophonic intent?

    True when any supplied key name contains ``homophon`` (e.g. the smoke's
    ``allow_homophones``) or the ``cipher_system`` hint mentions homophonic.
    """
    for key in (user_config or {}):
        if "homophon" in str(key).lower():
            return True
    cs = str((user_config or {}).get("cipher_system") or "").lower()
    return "homophon" in cs


def apply_config_defaults(exp_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """Merge the type defaults under ``config`` (config wins)."""
    merged = _config_defaults(exp_type)
    merged.update(config)
    return merged


def dedup_key(
    exp_type: str, language: str, effective_cipher: Any, config: dict[str, Any]
) -> str:
    """Semantic dedup key (A9).

    ``sha256`` of the type, language, the DERIVED effective cipher (token list +
    word lengths), and the defaults-applied config. Hashing the derived effective
    text (not branch name/key) makes dedup semantic; the defaults-applied config
    includes the resolved ``model_variant`` so dedup is variant-aware.
    """
    payload = {
        "type": exp_type,
        "language": language,
        "effective": {
            "tokens": list(effective_cipher.tokens),
            "word_lengths": [len(w) for w in effective_cipher.words],
        },
        "config": config,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# Record schema fields (plain dicts live in ``state.experiment_queue``).
def _new_record(
    *,
    exp_type: str,
    dedup: str,
    branch: str,
    snapshot: dict[str, Any],
    config: dict[str, Any],
    note: str,
    submitted_turn: int,
) -> dict[str, Any]:
    return {
        "experiment_id": uuid.uuid4().hex[:12],
        "dedup_key": dedup,
        "type": exp_type,
        "branch": branch,
        "snapshot": snapshot,
        "config": config,
        "note": note,
        "status": "pending",
        "submitted_turn": submitted_turn,
        "started_at": None,
        "completed_at": None,
        "elapsed_seconds": None,
        "inner_workers": None,
        "result": None,
        "error": None,
        "summary": "",
        "collected": False,
        "null_mask_session_id": None,
        "installed_as": None,
        "superseded_by": None,
        "orphan_reason": None,
    }


# ---------------------------------------------------------------------------
# Part 2 — arbiter math (A5)
# ---------------------------------------------------------------------------
_ARBITER_ENV_VARS = (
    "DECIPHER_HOMOPHONIC_PARALLEL_SEEDS",
    "DECIPHER_NULL_MASK_THREADS",
    "DECIPHER_TRANSFORM_RANK_THREADS",
)


def compute_arbiter(
    *, parallel_workers: str | None, experiment_slots: str | None, cpu_count: int
) -> tuple[int, int, int]:
    """Return (W, S, I). STATIC floor-division split; S x I <= W (A5).

    - ``W`` = int(DECIPHER_PARALLEL_WORKERS) if set, else max(1, cpu_count - 1).
    - ``S`` = clamp(int(DECIPHER_EXPERIMENT_SLOTS or 2), 1, W).
    - ``I`` = max(1, W // S).
    """
    cpu = max(1, int(cpu_count or 1))
    if parallel_workers is not None and str(parallel_workers).strip():
        try:
            w = int(parallel_workers)
        except (TypeError, ValueError):
            w = max(1, cpu - 1)
    else:
        w = max(1, cpu - 1)
    W = max(1, w)
    if experiment_slots is not None and str(experiment_slots).strip():
        try:
            s = int(experiment_slots)
        except (TypeError, ValueError):
            s = 2
    else:
        s = 2
    S = max(1, min(s, W))
    I = max(1, W // S)
    return W, S, I


def _experiment_sync_default() -> bool:
    return os.environ.get("DECIPHER_EXPERIMENT_SYNC", "").strip() in {
        "1", "true", "yes", "on",
    }


# ---------------------------------------------------------------------------
# Part 2 — the queue
# ---------------------------------------------------------------------------
class ExperimentQueue:
    """Daemon-thread execution + single-writer discipline + W/S/I arbiter (A5).

    ALL record mutations, env changes, and thread starts happen on the LEAD
    thread (in submit / poll / finalize). A worker thread writes ONLY into the
    private lock-guarded result box; ``poll`` harvests it. The queue object is
    NEVER serialized — it is constructed per ``run_v3`` call over the state
    records.
    """

    def __init__(
        self,
        *,
        synchronous: bool | None = None,
        workers: int | str | None = None,
        slots: int | str | None = None,
    ) -> None:
        self.synchronous = _experiment_sync_default() if synchronous is None else bool(synchronous)
        pw = str(workers) if workers is not None else os.environ.get("DECIPHER_PARALLEL_WORKERS")
        sl = str(slots) if slots is not None else os.environ.get("DECIPHER_EXPERIMENT_SLOTS")
        self.W, self.S, self.I = compute_arbiter(
            parallel_workers=pw, experiment_slots=sl, cpu_count=os.cpu_count() or 2
        )
        self._lock = threading.Lock()
        # experiment_id -> (result_dict | None, elapsed, exception | None)
        self._result_box: dict[str, tuple[Any, float, BaseException | None]] = {}
        self._threads: dict[str, threading.Thread] = {}
        # experiment_id -> Event set once the worker has written the box (test seam).
        self._settled: dict[str, threading.Event] = {}
        # Env-override bookkeeping (populated on the 0->1 running transition).
        self._saved_env: dict[str, str | None] | None = None
        # var -> True (arbiter overrode it) | False (user-set, kept).
        self._arbiter_overridden: dict[str, bool] = {}

    # --- arbiter env management (lead-thread only) --------------------------
    def _apply_arbiter_env(self) -> None:
        """0->1 running transition: save prior values, set overrides for vars the
        user did NOT explicitly set (user wins). Must run BEFORE Thread.start.

        Idempotent (F-1): if the overrides are already active (``_saved_env is not
        None`` <=> active, since ``_restore_arbiter_env`` nulls it), return without
        re-capturing priors. Without this guard, a single poll that harvests every
        running record and THEN promotes pending ones re-enters here with the
        overrides still in ``os.environ``, discarding the true priors (final
        restore would pin the three vars to ``I`` for the rest of the process) and
        flipping provenance to all-"user set" (false model-visible warnings)."""
        if self.synchronous:
            return
        if self._saved_env is not None:
            return
        self._saved_env = {}
        self._arbiter_overridden = {}
        for var in _ARBITER_ENV_VARS:
            present = var in os.environ
            self._saved_env[var] = os.environ.get(var)
            if present:
                self._arbiter_overridden[var] = False  # user wins
            else:
                os.environ[var] = str(self.I)
                self._arbiter_overridden[var] = True

    def _restore_arbiter_env(self) -> None:
        if self._saved_env is None:
            return
        for var, prior in self._saved_env.items():
            if prior is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prior
        self._saved_env = None
        self._arbiter_overridden = {}

    def _any_thread_alive(self) -> bool:
        with self._lock:
            threads = list(self._threads.values())
        return any(t.is_alive() for t in threads)

    def arbiter_status(self) -> dict[str, Any]:
        overridden = dict(self._arbiter_overridden)
        warnings: list[str] = []
        for var, did in overridden.items():
            if did is False:
                warnings.append(
                    f"{var} was set by the user ({os.environ.get(var)!r}); the "
                    f"arbiter did not override it — per-experiment inner "
                    f"parallelism may exceed the computed I={self.I}."
                )
        return {
            "W": self.W, "S": self.S, "I": self.I,
            "synchronous": self.synchronous,
            "arbiter_overridden": overridden,
            "warnings": warnings,
        }

    def finalize_env_restore(self) -> str | None:
        """Finalize env restore (F2): restore ONLY if no orphan thread is alive;
        otherwise leave the overrides in place (a live solver still reads them)
        and return a warning string for the artifact."""
        if self.synchronous or self._saved_env is None:
            return None
        if self._any_thread_alive():
            return (
                "Arbiter env overrides left in place at finalize: a background "
                "experiment thread is still alive and reading them."
            )
        self._restore_arbiter_env()
        return None

    # --- status helpers -----------------------------------------------------
    @staticmethod
    def _find(state: InvestigationState, experiment_id: str) -> dict[str, Any] | None:
        for record in reversed(state.experiment_queue):
            if record.get("experiment_id") == experiment_id:
                return record
        return None

    @staticmethod
    def _find_by_dedup(
        state: InvestigationState, key: str, statuses: set[str]
    ) -> dict[str, Any] | None:
        for record in reversed(state.experiment_queue):
            if record.get("dedup_key") == key and record.get("status") in statuses:
                return record
        return None

    def slots_summary(self, state: InvestigationState) -> dict[str, int]:
        running = sum(1 for r in state.experiment_queue if r.get("status") == "running")
        pending = sum(1 for r in state.experiment_queue if r.get("status") == "pending")
        return {"S": self.S, "I": self.I, "running": running, "pending": pending}

    # --- execution ----------------------------------------------------------
    def _worker(
        self,
        experiment_id: str,
        exp_type: str,
        cipher: Any,
        snapshot: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Worker-thread body: run the runner, write ONLY the result box."""
        start = time.time()
        result: Any = None
        exc: BaseException | None = None
        try:
            runner: Callable[..., dict[str, Any]] = EXPERIMENT_TYPES[exp_type]["runner"]
            result = runner(cipher, snapshot, config)
        except Exception as caught:  # noqa: BLE001 - failure becomes a record, not a crash
            exc = caught
        elapsed = time.time() - start
        with self._lock:
            self._result_box[experiment_id] = (result, elapsed, exc)
        settled = self._settled.get(experiment_id)
        if settled is not None:
            settled.set()

    def _start_thread(self, record: dict[str, Any], cipher: Any) -> None:
        experiment_id = record["experiment_id"]
        event = threading.Event()
        thread = threading.Thread(
            target=self._worker,
            args=(experiment_id, record["type"], cipher, record["snapshot"], record["config"]),
            daemon=True,
            name=f"experiment-{experiment_id}",
        )
        with self._lock:
            self._threads[experiment_id] = thread
            self._settled[experiment_id] = event
        thread.start()

    def _run_inline(
        self, record: dict[str, Any], state: InvestigationState, turn: int
    ) -> None:
        """Synchronous mode: run inline on the lead thread through the SAME record
        transitions; the arbiter env machinery is a no-op."""
        record["status"] = "running"
        record["started_at"] = time.time()
        record["inner_workers"] = self.I
        start = time.time()
        try:
            runner: Callable[..., dict[str, Any]] = EXPERIMENT_TYPES[record["type"]]["runner"]
            result = runner(state.cipher, record["snapshot"], record["config"])
            record["status"] = "completed"
            record["result"] = result
            record["summary"] = _completed_summary(record, result)
        except Exception as exc:  # noqa: BLE001
            record["status"] = "failed"
            record["error"] = _format_exception(exc)
            record["summary"] = _failed_summary(record, exc)
        record["completed_at"] = time.time()
        record["elapsed_seconds"] = round(time.time() - start, 3)
        _add_complete_evidence(state, record, turn)

    def _promote(self, state: InvestigationState, turn: int) -> None:
        """Promote pending -> running while slots are free (lead-thread only)."""
        if self.synchronous:
            pending = [r for r in state.experiment_queue if r.get("status") == "pending"]
            for record in pending:
                self._run_inline(record, state, turn)
            return
        running = [r for r in state.experiment_queue if r.get("status") == "running"]
        pending = [r for r in state.experiment_queue if r.get("status") == "pending"]
        free = self.S - len(running)
        to_start = pending[: max(0, free)] if free > 0 else []
        if not to_start:
            if not running:
                # Nothing running and nothing to start: ensure env is restored.
                self.finalize_env_restore()
            return
        had_running = bool(running)
        for record in to_start:
            record["status"] = "running"
            record["started_at"] = time.time()
            record["inner_workers"] = self.I
        # F2: the env set MUST complete before Thread.start().
        if not had_running:
            self._apply_arbiter_env()
        for record in to_start:
            self._start_thread(record, state.cipher)

    def submit(self, record: dict[str, Any], state: InvestigationState, turn: int) -> None:
        """Append a pending record and promote (starts it if a slot is free)."""
        state.experiment_queue.append(record)
        self._promote(state, turn)

    def poll(
        self,
        state: InvestigationState,
        turn: int | None = None,
        *,
        promote: bool = True,
    ) -> list[str]:
        """Harvest the result box on the lead thread; flip running -> completed/
        failed; stamp elapsed/summary; append experiment_complete evidence; then
        (unless ``promote=False``) promote pending into freed slots.

        Returns the ids that transitioned to completed/failed this call.
        """
        t = turn if turn is not None else state.turn
        with self._lock:
            harvested = dict(self._result_box)
            self._result_box.clear()
        transitioned: list[str] = []
        for experiment_id, (result, elapsed, exc) in harvested.items():
            record = self._find(state, experiment_id)
            if record is None or record.get("status") != "running":
                continue
            # F-3: stamp completed_at finish-true (started_at + measured elapsed),
            # NOT time.time() at poll — the lead may poll long after the worker
            # finished, and the overlap-gate evidence must reflect the real finish.
            started = record.get("started_at")
            record["completed_at"] = (
                (started + elapsed) if isinstance(started, (int, float))
                else time.time()
            )
            record["elapsed_seconds"] = round(elapsed, 3)
            if exc is not None:
                record["status"] = "failed"
                record["error"] = _format_exception(exc)
                record["summary"] = _failed_summary(record, exc)
            else:
                record["status"] = "completed"
                record["result"] = result
                record["summary"] = _completed_summary(record, result)
            with self._lock:
                self._threads.pop(experiment_id, None)
            _add_complete_evidence(state, record, t)
            transitioned.append(experiment_id)
        if promote:
            self._promote(state, t)
        elif not any(r.get("status") == "running" for r in state.experiment_queue):
            self.finalize_env_restore()
        return transitioned

    def wait_settled(self, experiment_id: str, timeout: float | None = None) -> bool:
        """Block until a worker has written the box for ``experiment_id`` (test
        seam: Events only, no sleeps)."""
        event = self._settled.get(experiment_id)
        if event is None:
            return True
        return event.wait(timeout)


# ---------------------------------------------------------------------------
# Summaries + evidence (lead-side)
# ---------------------------------------------------------------------------
def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _completed_summary(record: dict[str, Any], result: dict[str, Any]) -> str:
    status = result.get("status")
    solver = result.get("solver")
    preview = str(result.get("final_decryption") or "").replace("\n", " ")[:160]
    text = (
        f"{record['type']} {status} solver={solver} "
        f"elapsed={record.get('elapsed_seconds')}s :: {preview}"
    )
    return text[:400]


def _failed_summary(record: dict[str, Any], exc: BaseException) -> str:
    return f"{record['type']} failed: {_format_exception(exc)}"[:400]


def _add_complete_evidence(
    state: InvestigationState, record: dict[str, Any], turn: int
) -> None:
    state.add_evidence(
        "experiment_complete",
        turn=int(turn or 0),
        summary=record.get("summary") or "",
        experiment_id=record.get("experiment_id"),
        type=record.get("type"),
        status=record.get("status"),
        elapsed_seconds=record.get("elapsed_seconds"),
    )


# ---------------------------------------------------------------------------
# Part 3 — lead tools
# ---------------------------------------------------------------------------
EXPERIMENT_SUBMIT_TOOL = {
    "name": "experiment_submit",
    "description": (
        "Queue a long-running automated-solver experiment on a branch; it runs "
        "in the BACKGROUND while you keep working, and you adjudicate it later "
        "with experiment_collect (collect/install the result rather than "
        "submitting duplicate jobs). Type `automated_solver` runs the no-LLM "
        "solver stack (homophonic anneal, transform screens via transform_search, "
        "null-mask bakeoffs via homophonic_refinement=null_masks). Type "
        "`quagmire3_shotgun` runs the Rust keyed-tableau/Quagmire III shotgun "
        "search (use when period evidence suggests a keyed tableau; results "
        "install as decoded branches via experiment_collect). `config` is "
        "typed per experiment type (see its schema); its keys are the ONLY "
        "supported controls. Do NOT set `language` — it is host-derived from the "
        "run. Select homophonic behavior via the homophonic_* fields, not "
        "free-form flags. Pass `resubmit=<experiment_id>` to re-run an orphaned/"
        "failed experiment from its stored snapshot (works even after resume). "
        "Returns the experiment id and queue slots; duplicate specs dedup to a "
        "prior completed run for free. An invalid config returns a structured "
        "error with a corrected, family-consistent example to copy."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "description": (
                    "Experiment type. Supported: 'automated_solver' (the no-LLM "
                    "solver stack) and 'quagmire3_shotgun' (Rust Quagmire III "
                    "keyed-tableau search). Each type's `config` is validated "
                    "against its own schema."
                ),
            },
            "branch": {
                "type": "string",
                "description": (
                    "Branch to run the experiment on (required unless `resubmit`)."
                ),
            },
            "config": {
                "anyOf": [
                    {
                        "title": "automated_solver config",
                        **model_facing_config_schema("automated_solver"),
                    },
                    {
                        "title": "quagmire3_shotgun config",
                        **model_facing_config_schema("quagmire3_shotgun"),
                    },
                ],
                "description": (
                    "Per-type config; see the branch matching your `type`."
                ),
            },
            "note": {"type": "string"},
            "resubmit": {
                "type": "string",
                "description": (
                    "Re-run an orphaned/failed experiment id from its stored "
                    "snapshot (mutually exclusive with `branch`/`config`)."
                ),
            },
        },
        "required": ["type"],
    },
}

EXPERIMENT_COLLECT_TOOL = {
    "name": "experiment_collect",
    "description": (
        "Adjudicate queued experiments. With no id: poll the queue and return one "
        "status line per outstanding experiment plus full summaries of any that "
        "just completed. With `experiment_id`: return the result packet (solver, "
        "decoded preview, route/primary step) and mark it collected; pass "
        "`install=true` (completed runs only) to adopt the solved branch under a "
        "fresh name (`as_name` optional), optionally choosing which ranked "
        "finalist via `candidate_rank` (quagmire3_shotgun). A null-mask bakeoff "
        "exposes a finalist review session in the packet."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "experiment_id": {"type": "string"},
            "install": {"type": "boolean"},
            "as_name": {"type": "string"},
            "candidate_rank": {
                "type": "integer",
                "description": (
                    "1-based finalist to install for ranked-result experiments "
                    "(quagmire3_shotgun). Default 1 = best."
                ),
            },
        },
    },
}

# The final V3_LEAD_TOOL_DEFINITIONS is assembled in loop_v3 (F7) — never
# appended into episodes.py's constant (import cycle). loop_v3 concatenates
# these onto the episode-level list.
EXPERIMENT_TOOL_DEFINITIONS = [EXPERIMENT_SUBMIT_TOOL, EXPERIMENT_COLLECT_TOOL]
EXPERIMENT_TOOL_NAMES = frozenset({"experiment_submit", "experiment_collect"})


def _resolve_variant(language: str, variant: str | None) -> dict[str, Any] | None:
    """Validate a variant at SUBMIT time on the lead thread (F1b). Returns a
    structured error dict on failure, or ``None`` when valid (incl. None =
    default)."""
    if variant is None:
        return None
    try:
        model_registry.resolve_language_model(language, variant=variant)
    except model_registry.ModelVariantError as exc:
        return {
            "error": str(exc),
            "requested_model_variant": variant,
            "available_model_variants": list(exc.available),
        }
    return None


def dispatch_experiment_submit(
    queue: ExperimentQueue,
    state: InvestigationState,
    workspace: Workspace,
    executor: Any,
    args: dict[str, Any],
    turn: int,
) -> dict[str, Any]:
    exp_type = str(args.get("type") or "")
    if exp_type not in EXPERIMENT_TYPES:
        return {
            "error": f"unknown experiment type: {exp_type!r}",
            "available_types": sorted(EXPERIMENT_TYPES),
        }

    resubmit_id = args.get("resubmit")
    if resubmit_id:
        return _dispatch_resubmit(queue, state, str(resubmit_id), turn)

    branch = str(args.get("branch") or "")
    if not branch:
        return {"error": "experiment_submit requires `branch` (or `resubmit`)."}
    if not workspace.has_branch(branch):
        return {"error": f"unknown branch: {branch!r}"}

    user_config = dict(args.get("config") or {})
    config_errors = validate_experiment_config(exp_type, user_config)
    if config_errors:
        # Slice 5: a structured validation error carries a VALID corrected
        # example (family-consistent) plus a note so the model can fix its call
        # without guessing. Also surface the supported-key contract inline.
        return {
            "error": "invalid experiment config",
            "config_errors": config_errors,
            "corrected_example": corrected_config_example(exp_type, user_config),
            "config_schema": model_facing_config_schema(exp_type),
            "note": (
                "`language` is host-derived and must not be supplied; select "
                "homophonic behavior via the homophonic_* fields (not free-form "
                "flags like allow_homophones); runtime controls such as "
                "max_runtime_seconds are not accepted. Copy `corrected_example`."
            ),
        }

    # F1b: resolve model_variant at submit on the lead thread; default from the
    # lead executor's current selection.
    if "model_variant" in user_config:
        variant = user_config.get("model_variant")
    else:
        variant = getattr(executor, "_model_variant", None)
    variant_error = _resolve_variant(state.language, variant)
    if variant_error is not None:
        return variant_error

    config = apply_config_defaults(exp_type, user_config)
    config["model_variant"] = variant
    config["language"] = state.language

    snapshot = copy.deepcopy(_serialize_branch(workspace, branch))
    effective = _reconstruct_effective(state.cipher, snapshot)
    key = dedup_key(exp_type, state.language, effective, config)

    # Dedup precedence (A9): completed -> deduplicated; pending|running ->
    # already_queued; orphaned -> new record + stamp superseded_by.
    completed = queue._find_by_dedup(state, key, {"completed"})
    if completed is not None:
        return {
            "experiment_id": completed["experiment_id"],
            "status": "completed",
            "summary": completed.get("summary") or "",
            "deduplicated": True,
            "slots": queue.slots_summary(state),
        }
    active = queue._find_by_dedup(state, key, {"pending", "running"})
    if active is not None:
        return {
            "experiment_id": active["experiment_id"],
            "status": active["status"],
            "already_queued": True,
            "slots": queue.slots_summary(state),
        }
    orphaned = queue._find_by_dedup(state, key, {"orphaned"})

    record = _new_record(
        exp_type=exp_type,
        dedup=key,
        branch=branch,
        snapshot=snapshot,
        config=config,
        note=str(args.get("note") or ""),
        submitted_turn=turn,
    )
    queue.submit(record, state, turn)
    if orphaned is not None:
        orphaned["superseded_by"] = record["experiment_id"]

    response = {
        "experiment_id": record["experiment_id"],
        "status": record["status"],
        "slots": queue.slots_summary(state),
        "arbiter": queue.arbiter_status(),
    }
    warnings = queue.arbiter_status().get("warnings") or []
    if warnings:
        response["warnings"] = warnings
    return response


def _dispatch_resubmit(
    queue: ExperimentQueue, state: InvestigationState, resubmit_id: str, turn: int
) -> dict[str, Any]:
    source = queue._find(state, resubmit_id)
    if source is None:
        return {"error": f"unknown experiment_id for resubmit: {resubmit_id!r}"}
    dedup = source.get("dedup_key") or ""
    # F6: dedup-check against same-key completed records FIRST (cheap path). A
    # completed source is caught here (it is its own same-key completed record).
    completed = queue._find_by_dedup(state, dedup, {"completed"})
    if completed is not None:
        return {
            "experiment_id": completed["experiment_id"],
            "status": "completed",
            "summary": completed.get("summary") or "",
            "deduplicated": True,
            "slots": queue.slots_summary(state),
        }
    # F-2: a same-key pending|running record means the work is already live —
    # return already_queued (mirrors the submit path) instead of duplicating it
    # and stamping a still-running record superseded_by. This also catches a
    # pending|running SOURCE (it is its own same-key active record).
    active = queue._find_by_dedup(state, dedup, {"pending", "running"})
    if active is not None:
        return {
            "experiment_id": active["experiment_id"],
            "status": active["status"],
            "already_queued": True,
            "slots": queue.slots_summary(state),
        }
    # F-2: only orphaned/failed experiments can be re-run from their snapshot.
    if source.get("status") not in {"orphaned", "failed"}:
        return {
            "error": (
                f"cannot resubmit experiment {resubmit_id!r} with status "
                f"{source.get('status')!r}; only orphaned or failed experiments "
                "can be re-run."
            ),
        }
    record = _new_record(
        exp_type=source["type"],
        dedup=source.get("dedup_key") or "",
        branch=source.get("branch") or "",
        snapshot=copy.deepcopy(source.get("snapshot") or {}),
        config=copy.deepcopy(source.get("config") or {}),
        note=source.get("note") or "",
        submitted_turn=turn,
    )
    queue.submit(record, state, turn)
    source["superseded_by"] = record["experiment_id"]
    return {
        "experiment_id": record["experiment_id"],
        "status": record["status"],
        "resubmitted_from": resubmit_id,
        "slots": queue.slots_summary(state),
        "arbiter": queue.arbiter_status(),
    }


def _status_line(record: dict[str, Any]) -> str:
    eid = record.get("experiment_id")
    etype = record.get("type")
    status = record.get("status")
    branch = record.get("branch")
    elapsed = record.get("elapsed_seconds")
    el = f" {elapsed}s" if isinstance(elapsed, (int, float)) else ""
    line = f"{eid} {etype} [{status}] branch={branch}{el}"
    if status == "completed" and record.get("summary"):
        line += f" — {record['summary']}"
    return line


def _route_and_primary_steps(result: dict[str, Any]) -> tuple[Any, Any]:
    steps = result.get("steps") or []
    route_step = steps[0] if steps else None
    primary_step = next(
        (s for s in steps if isinstance(s, dict) and s.get("name") != "route_automated_solver"),
        route_step,
    )
    return route_step, primary_step


def _null_mask_step(result: dict[str, Any]) -> dict[str, Any] | None:
    for step in result.get("steps") or []:
        if (
            isinstance(step, dict)
            and step.get("name") == "search_null_masks"
            and step.get("status") == "completed"
        ):
            return step
    return None


def _mirror_null_mask_metadata(
    workspace: Workspace, name: str, result: dict[str, Any]
) -> None:
    """Mirror the v2 `null_mask_selected` metadata block onto the installed
    branch (quote-located from _tool_search_automated_solver)."""
    step = _null_mask_step(result)
    if step is None:
        return
    selected = step.get("selected")
    if not (isinstance(selected, dict) and selected.get("decryption")):
        return
    branch = workspace.get_branch(name)
    branch.metadata.update({
        "decoded_text": str(selected.get("decryption") or ""),
        "decoded_text_source": "experiment_collect:null_masks",
        "cipher_mode": branch.metadata.get("cipher_mode") or "homophonic_substitution",
        "key_type": "homophonic_with_null_mask",
        "candidate_renderer": "key_with_null_mask_v1",
        "candidate_capabilities": [
            "editable_key", "editable_null_mask", "editable_boundaries",
        ],
        "null_mask_selected": {
            "mask": selected.get("mask") or [],
            "validation_score_v2": selected.get("validation_score_v2"),
            "confirmed_validation_score_v2": selected.get("confirmed_validation_score_v2"),
            "selection_score": selected.get("selection_score"),
            "consensus_polish": selected.get("consensus_polish"),
            "confirmation": selected.get("confirmation"),
        },
    })
    if "null_mask_finalist" not in branch.tags:
        branch.tags.append("null_mask_finalist")


def _install_experiment_branch(
    workspace: Workspace,
    record: dict[str, Any],
    as_name: str | None,
    turn: int,
    candidate_rank: int = 1,
) -> str:
    """Deep-copy the stored snapshot, restore under a fresh name, set the result
    key, created_iteration = turn; mirror null-mask metadata. Source branch never
    mutated (unlike the v2 in-place tool). ``candidate_rank`` is accepted for
    installer-seam uniformity and ignored (this type has no ranked finalists —
    dispatch rejects a supplied candidate_rank for it before calling here)."""
    snap = copy.deepcopy(record.get("snapshot") or {})
    result = record.get("result") or {}
    key = {int(k): int(v) for k, v in (result.get("key") or {}).items()}
    target = str(as_name) if as_name else f"exp_{str(record['experiment_id'])[:6]}_{record.get('branch')}"
    name = target
    suffix = 2
    while workspace.has_branch(name):
        name = f"{target}_{suffix}"
        suffix += 1
    install_snap = dict(snap)
    install_snap["name"] = name
    install_snap["created_iteration"] = turn
    _restore_branch_into(workspace, install_snap)
    workspace.set_full_key(name, key)
    _mirror_null_mask_metadata(workspace, name, result)
    return name


def _install_quagmire3_branch(
    workspace: Workspace,
    record: dict[str, Any],
    as_name: str | None,
    turn: int,
    candidate_rank: int = 1,
) -> str | dict[str, Any]:
    """Install a quagmire3 shotgun finalist as a mode-specific decoded branch,
    mirroring the v2 tool block (tools_v2.py:7229-7298) through the experiment-
    snapshot path (the source branch is never mutated). Does NOT set_full_key:
    the inherited snapshot key is left as-is, and ``_decoded_text_for_panel``
    prefers ``metadata['decoded_text']`` — which is what attestation/verify/
    declare hash, making the gate reachable. Returns the installed branch name,
    or a structured ``{"error": ...}`` dict."""
    result = record.get("result") or {}
    candidates = result.get("top_candidates") or []
    if not candidates:
        return {"error": "experiment produced no quagmire3 candidates to install"}
    if candidate_rank < 1 or candidate_rank > len(candidates):
        return {
            "error": (
                f"candidate_rank {candidate_rank} out of range; valid "
                f"1..{len(candidates)}"
            )
        }
    candidate = candidates[candidate_rank - 1]
    if not str(candidate.get("plaintext") or "").strip():
        # An empty decoded_text would make _decoded_text_for_panel fall back to
        # the inherited snapshot key — a stale decode presented as the quagmire
        # branch text.
        return {
            "error": (
                f"selected candidate (rank {candidate_rank}) has no plaintext; "
                "it cannot be installed as a decoded branch"
            )
        }
    metadata = dict(candidate.get("metadata") or {})
    alphabet_keyword = str(metadata.get("alphabet_keyword") or "unknown")
    cycleword = str(metadata.get("cycleword") or candidate.get("key") or "unknown")

    target = str(as_name) if as_name else f"exp_{str(record['experiment_id'])[:6]}_{record.get('branch')}"
    name = target
    suffix = 2
    while workspace.has_branch(name):
        name = f"{target}_{suffix}"
        suffix += 1
    install_snap = dict(copy.deepcopy(record.get("snapshot") or {}))
    install_snap["name"] = name
    install_snap["created_iteration"] = turn
    _restore_branch_into(workspace, install_snap)

    step = (result.get("steps") or [{}])[0]
    branch = workspace.get_branch(name)
    # A null-mask block inherited from the source snapshot would shadow
    # decoded_text in _metadata_decoded_text (mask+key render wins), so the
    # verify/declare hash would bind the OLD null-mask decode, not this result.
    branch.metadata.pop("null_mask_finalist", None)
    branch.metadata.pop("null_mask_selected", None)
    branch.metadata.update({
        "cipher_mode": "quagmire3",
        "mode_status": "active",
        "mode_confidence": "medium",
        "mode_evidence": (
            f"Installed by experiment_collect from quagmire3_shotgun experiment "
            f"{record['experiment_id']} (alphabet keyword {alphabet_keyword!r}, "
            f"cycleword {cycleword!r}, rank {candidate_rank})."
        ),
        "mode_counter_evidence": (
            "Bounded shotgun search can overfit; verify readability before "
            "declaring."
        ),
        "key_type": "QuagmireKey",
        "quagmire_type": metadata.get("quagmire_type", "quag3"),
        "alphabet_keyword": alphabet_keyword,
        "cycleword": cycleword,
        "cycleword_shifts": metadata.get("cycleword_shifts", candidate.get("shifts")),
        "plaintext_alphabet": metadata.get("plaintext_alphabet"),
        "ciphertext_alphabet": metadata.get("ciphertext_alphabet"),
        "quagmire_score": candidate.get("score"),
        "quagmire_selection_score": candidate.get("selection_score"),
        "decoded_text": candidate.get("plaintext", ""),
        "decoded_text_source": "experiment_collect:quagmire3_shotgun",
        "search_metadata": {
            "solver": "quagmire3_shotgun_rust",
            "engine": "rust_shotgun",
            "experiment_id": record["experiment_id"],
            "candidate_rank": candidate_rank,
            "hillclimbs": step.get("hillclimbs"),
            "restarts": step.get("restarts"),
            "keyword_lengths": step.get("keyword_lengths"),
            "cycleword_lengths": step.get("cycleword_lengths"),
            "nominal_proposals": result.get("nominal_proposals"),
        },
    })
    workspace.tag(name, "hypothesis")
    workspace.tag(name, "mode:quagmire3")
    workspace.tag(name, "mode:keyed_tableau_polyalphabetic")
    return name


# Bind the ranked-finalist installer now that it is defined (declared in the
# registry above with a None placeholder to avoid a forward reference).
EXPERIMENT_TYPES["quagmire3_shotgun"]["installer"] = _install_quagmire3_branch


def _finalist_source_branch(
    workspace: Workspace, record: dict[str, Any], installed_name: str | None
) -> str:
    """F7 rule: the record's source branch if it still exists, else the freshly
    installed `exp_` branch when install=true; else the (dead) source branch."""
    src = str(record.get("branch") or "")
    if workspace.has_branch(src):
        return src
    if installed_name and workspace.has_branch(installed_name):
        return installed_name
    return src


def _maybe_create_null_mask_session(
    executor: Any,
    workspace: Workspace,
    record: dict[str, Any],
    installed_name: str | None,
) -> str | None:
    """Create the state-owned null-mask finalist session at FIRST collect of a
    result carrying a completed search_null_masks step; the record's
    null_mask_session_id prevents duplicates."""
    result = record.get("result") or {}
    step = _null_mask_step(result)
    if step is None:
        return None
    if record.get("null_mask_session_id"):
        return record["null_mask_session_id"]
    source_branch = _finalist_source_branch(workspace, record, installed_name)
    session_id = executor._new_null_mask_session(source_branch=source_branch, result=step)
    record["null_mask_session_id"] = session_id
    return session_id


def dispatch_experiment_collect(
    queue: ExperimentQueue,
    state: InvestigationState,
    workspace: Workspace,
    executor: Any,
    args: dict[str, Any],
    turn: int,
) -> dict[str, Any]:
    experiment_id = args.get("experiment_id")

    # No id: poll + queue status + full summaries of newly completed ones.
    if not experiment_id:
        transitioned = queue.poll(state, turn)
        outstanding = [
            _status_line(r) for r in state.experiment_queue if not r.get("collected")
        ]
        newly_completed = [
            {
                "experiment_id": r["experiment_id"],
                "type": r.get("type"),
                "status": r.get("status"),
                "summary": r.get("summary") or "",
            }
            for r in state.experiment_queue
            if r.get("experiment_id") in transitioned
        ]
        return {
            "queue": outstanding,
            "newly_completed": newly_completed,
            "slots": queue.slots_summary(state),
        }

    # With id: refresh via poll first so the record reflects the latest box.
    queue.poll(state, turn)
    record = queue._find(state, str(experiment_id))
    if record is None:
        return {"error": f"unknown experiment_id: {experiment_id!r}"}

    status = record.get("status")
    if status in {"pending", "running"}:
        return {
            "experiment_id": record["experiment_id"],
            "type": record.get("type"),
            "status": status,
            "note": "still running; poll again with experiment_collect (no id).",
            "slots": queue.slots_summary(state),
        }

    install = bool(args.get("install"))
    result = record.get("result") or {}
    route_step, primary_step = _route_and_primary_steps(result)

    if status in {"failed", "orphaned"}:
        record["collected"] = True
        return {
            "experiment_id": record["experiment_id"],
            "type": record.get("type"),
            "status": status,
            "config": record.get("config"),
            "error": record.get("error"),
            "orphan_reason": record.get("orphan_reason"),
            "note": (
                "orphaned/failed experiments are not installable; resubmit with "
                "experiment_submit(resubmit=<id>) to re-run from its snapshot."
            ),
        }

    # Completed.
    candidate_rank_arg = args.get("candidate_rank")
    candidate_rank = max(1, int(candidate_rank_arg or 1))
    # candidate_rank is only meaningful for ranked-finalist types; supplying it
    # for any other type is a usage error (not silently ignored).
    if candidate_rank_arg is not None and record.get("type") != "quagmire3_shotgun":
        return {
            "experiment_id": record["experiment_id"],
            "type": record.get("type"),
            "status": status,
            "error": (
                "candidate_rank is only supported for experiment types with "
                "ranked finalists (quagmire3_shotgun)"
            ),
        }

    installed_name: str | None = None
    if install:
        installer = (
            EXPERIMENT_TYPES.get(record["type"], {}).get("installer")
            or _install_experiment_branch
        )
        installed = installer(
            workspace, record, args.get("as_name"), turn,
            candidate_rank=candidate_rank,
        )
        if isinstance(installed, dict) and "error" in installed:
            # A structured installer error (no candidates / rank out of range):
            # adjudicate once (mark collected) and surface it, no branch created.
            record["collected"] = True
            return {
                "experiment_id": record["experiment_id"],
                "type": record.get("type"),
                "status": status,
                "error": installed["error"],
            }
        installed_name = installed
        record["installed_as"] = installed_name
        source_card = state.hypothesis_board.get(str(record.get("branch") or ""))
        if source_card is not None:
            fields = {
                key: source_card.get(key)
                for key in CARD_MIRROR_KEYS
                if source_card.get(key) is not None
            }
            fields["mode_status"] = "active"
            fields["evidence_source"] = f"experiment:{record['experiment_id']}"
            fields["mode_evidence"] = str(record.get("summary") or "experiment result")
            state.hypothesis_board.update(workspace, installed_name, **fields)

    session_id = _maybe_create_null_mask_session(
        executor, workspace, record, installed_name
    )

    record["collected"] = True
    packet: dict[str, Any] = {
        "experiment_id": record["experiment_id"],
        "type": record.get("type"),
        "status": status,
        "config": record.get("config"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "inner_workers": record.get("inner_workers"),
        "solver": result.get("solver"),
        "decoded_preview": str(result.get("final_decryption") or "")[:800],
        "route_step": _strip_packet_keys(route_step),
        "primary_step": _strip_packet_keys(primary_step),
        "error": result.get("error_message") or None,
    }
    if installed_name is not None:
        packet["installed_as"] = installed_name
    # Ranked-finalist summary (generic: any result carrying top_candidates) so a
    # lead can pick a candidate_rank without a second tool call. Previews only —
    # decoded_preview already carries the best candidate's full-ish text.
    top_candidates = result.get("top_candidates")
    if top_candidates:
        packet["candidates"] = [
            {
                "rank": c.get("rank", idx),
                "score": c.get("score"),
                "selection_score": c.get("selection_score"),
                "preview": c.get("preview"),
            }
            for idx, c in enumerate(top_candidates, 1)
        ]
    if session_id is not None:
        # F7: surface the session so the lead can find it — id + capped initial
        # review + the v2 suggested-next-tools triplet.
        packet["null_mask_search_session_id"] = session_id
        packet["null_mask_finalist_review"] = executor._null_mask_finalist_review(
            session_id=session_id,
            start_rank=1,
            count=12,
            review_chars=1000,
            good_score_gap=0.25,
        )
        packet["suggested_next_tools"] = [
            "search_review_null_mask_finalists",
            "act_rate_null_mask_finalist",
            "act_install_null_mask_finalists",
        ]
        packet["null_mask_note"] = (
            "A null-mask finalist menu is available; installing finalists from "
            "the session requires its source branch to be live."
        )
    return packet
