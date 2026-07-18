"""Experiment-queue tests (M4). Fakes + local compute only; no live LLM."""
from __future__ import annotations

import copy
import json
import os
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from agent.tools_v2 import NoGatesPolicy, WorkspaceToolExecutor
from analysis import model_registry
from investigation import experiments as exp
from investigation.experiments import (
    EXPERIMENT_SUBMIT_TOOL,
    EXPERIMENT_TYPES,
    ExperimentQueue,
    apply_config_defaults,
    compute_arbiter,
    corrected_config_example,
    dedup_key,
    dispatch_experiment_collect,
    dispatch_experiment_submit,
    model_facing_config_schema,
    register_experiment_type,
    validate_experiment_config,
)
from investigation.state import InvestigationState
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _caesar(text: str, shift: int) -> str:
    return "".join(
        chr((ord(c) - 65 + shift) % 26 + 65) if c.isalpha() else c for c in text
    )


def _make_state(plaintext: str = "THE QUICK BROWN FOX", shift: int = 3):
    raw = _caesar(plaintext, shift)
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ws = Workspace(cipher_text=ct)
    state = InvestigationState(workspace=ws, language="en")
    return ct, state


def _fake_executor(model_variant=None):
    return SimpleNamespace(_model_variant=model_variant)


@contextmanager
def _registered_type(name, entry):
    register_experiment_type(name, entry)
    try:
        yield
    finally:
        EXPERIMENT_TYPES.pop(name, None)


def _blocking_type(gates):
    """A fake type whose runner blocks on a per-experiment threading.Event keyed
    by ``config['gate_id']`` (Events only, no sleeps)."""
    def runner(cipher, snapshot, config):
        gate = gates.get(config.get("gate_id"))
        if gate is not None:
            gate.wait()
        return {"status": "completed", "gate_id": config.get("gate_id"),
                "final_decryption": "FAKE"}
    return {
        "config_schema": {
            "type": "object",
            "properties": {"gate_id": {"type": "string"}},
        },
        "config_defaults": {"gate_id": None},
        "runner": runner,
        "description": "blocking fake",
    }


# ---------------------------------------------------------------------------
# Arbiter math (Part 2 / Part 7)
# ---------------------------------------------------------------------------
def test_arbiter_math_env_unset():
    # W = max(1, cpu-1); S = clamp(2,1,W); I = W//S.
    assert compute_arbiter(parallel_workers=None, experiment_slots=None, cpu_count=8) == (7, 2, 3)
    assert compute_arbiter(parallel_workers=None, experiment_slots=None, cpu_count=2) == (1, 1, 1)


def test_arbiter_math_env_set():
    assert compute_arbiter(parallel_workers="4", experiment_slots="2", cpu_count=99) == (4, 2, 2)
    assert compute_arbiter(parallel_workers="6", experiment_slots="3", cpu_count=2) == (6, 3, 2)


def test_arbiter_math_slots_gt_workers_clamped():
    # S > W clamps to W; I = W // S = 1.
    assert compute_arbiter(parallel_workers="4", experiment_slots="9", cpu_count=8) == (4, 4, 1)


def test_arbiter_math_w_one():
    assert compute_arbiter(parallel_workers="1", experiment_slots="2", cpu_count=8) == (1, 1, 1)


# ---------------------------------------------------------------------------
# Registry validation — structured, never raised (Part 1 / Part 7)
# ---------------------------------------------------------------------------
def test_unknown_type_is_structured_error():
    _ct, state = _make_state()
    out = dispatch_experiment_submit(
        ExperimentQueue(synchronous=True), state, state.workspace,
        _fake_executor(), {"type": "nope", "branch": "main"}, 1,
    )
    assert "error" in out and "automated_solver" in out["available_types"]


def test_unknown_config_key_rejected():
    errors = validate_experiment_config("automated_solver", {"bogus": 1})
    assert errors and any("bogus" in e for e in errors)


def test_enum_violation_rejected_and_promote_excluded():
    assert validate_experiment_config("automated_solver", {"transform_search": "promote"})
    assert validate_experiment_config("automated_solver", {"homophonic_budget": "wild"})
    # valid values pass.
    assert validate_experiment_config("automated_solver", {"transform_search": "screen"}) == []


def test_submit_config_error_is_structured_not_raised():
    _ct, state = _make_state()
    out = dispatch_experiment_submit(
        ExperimentQueue(synchronous=True), state, state.workspace, _fake_executor(),
        {"type": "automated_solver", "branch": "main", "config": {"transform_search": "promote"}}, 1,
    )
    assert "config_errors" in out


# ---------------------------------------------------------------------------
# Slice 5 — typed experiment schema + alternate-search config (M5.3)
# ---------------------------------------------------------------------------
def test_model_facing_config_schema_shape():
    """The provider-visible `config` schema exposes the real keys/enums/defaults,
    folds in the universal `model_variant`, and is host-language-firewalled."""
    config_schema = EXPERIMENT_SUBMIT_TOOL["input_schema"]["properties"]["config"]
    # `config` is an anyOf union of the two per-type schemas (each titled).
    auto_branch = next(
        b for b in config_schema["anyOf"] if b.get("title") == "automated_solver config"
    )
    schema = {k: v for k, v in auto_branch.items() if k != "title"}
    # Same object the builder returns for automated_solver.
    assert schema == model_facing_config_schema("automated_solver")
    assert schema["type"] == "object"
    # Provider-side unknown-key rejection.
    assert schema["additionalProperties"] is False
    props = schema["properties"]
    # language is host-derived: it must NOT be a supplied key.
    assert "language" not in props
    # Universal key folded in.
    assert "model_variant" in props
    # Real enums exposed (homophonic behavior selected via these fields).
    assert props["homophonic_solver"]["enum"] == ["zenith_native", "legacy"]
    assert set(props["homophonic_refinement"]["enum"]) >= {
        "none", "targeted_repair", "null_masks",
    }
    # `promote` is deliberately absent from transform_search (F5 purity break).
    assert "promote" not in props["transform_search"]["enum"]
    # Defaults + concise descriptions are advertised.
    assert props["homophonic_solver"]["default"] == "zenith_native"
    assert props["transform_search"]["default"] == "off"
    assert all(props[k].get("description") for k in props)
    # Nullable fields keep the local-validator type-list dialect.
    assert props["transform_search_max_generated_candidates"]["type"] == ["integer", "null"]
    assert props["model_variant"]["type"] == ["string", "null"]


def test_advertised_schema_matches_validated_contract():
    """Every advertised key/enum is exactly what dispatch validation enforces —
    no drift between what the model is told and what it is held to."""
    advertised = model_facing_config_schema("automated_solver")["properties"]
    for key, sub in exp._AUTOMATED_SOLVER_SCHEMA["properties"].items():
        assert key in advertised
        assert advertised[key]["type"] == sub["type"]
        if "enum" in sub:
            assert advertised[key]["enum"] == sub["enum"]
    # The advertised key set equals validated (schema props) ∪ universal keys.
    allowed = set(exp._AUTOMATED_SOLVER_SCHEMA["properties"]) | {"model_variant"}
    assert set(advertised) == allowed


@pytest.mark.parametrize(
    "bad_key", ["target_language", "allow_homophones", "max_runtime_seconds", "language"]
)
def test_provider_schema_rejects_unknown_config_keys_before_dispatch(bad_key):
    """The M5.2 smoke guessed target_language/allow_homophones/max_runtime_seconds;
    plus the host-derived `language`. Each is rejected BEFORE a queue record is
    created, and the error carries a valid corrected example + schema."""
    _ct, state = _make_state()
    q = ExperimentQueue(synchronous=True)
    out = dispatch_experiment_submit(
        q, state, state.workspace, _fake_executor(),
        {"type": "automated_solver", "branch": "main", "config": {bad_key: "x"}}, 1,
    )
    assert out["error"] == "invalid experiment config"
    assert any(bad_key in e for e in out["config_errors"])
    # Rejected before dispatch: nothing was queued.
    assert state.experiment_queue == []
    # Structured recovery: a valid corrected example + the schema contract.
    assert validate_experiment_config("automated_solver", out["corrected_example"]) == []
    assert out["config_schema"]["additionalProperties"] is False
    assert "language" in out["note"]


def test_validation_error_returns_valid_corrected_example():
    """A structured validation error yields a valid, family-consistent example."""
    # Homophonic intent (allow_homophones flag) → corrected example uses the
    # supported homophonic_* mechanism, not the guessed flag.
    homo = corrected_config_example(
        "automated_solver",
        {"allow_homophones": True, "target_language": "de", "max_runtime_seconds": 900},
    )
    assert validate_experiment_config("automated_solver", homo) == []
    assert homo.get("cipher_system") == "homophonic_substitution"
    assert homo.get("homophonic_solver") == "zenith_native"
    assert "allow_homophones" not in homo and "target_language" not in homo

    # A supported key with a bad enum value is dropped so the default applies.
    fixed = corrected_config_example("automated_solver", {"transform_search": "promote"})
    assert validate_experiment_config("automated_solver", fixed) == []
    assert "transform_search" not in fixed  # invalid value dropped → default

    # A supported key with a valid value is preserved verbatim.
    kept = corrected_config_example(
        "automated_solver", {"cipher_system": "vigenere", "bogus": 1}
    )
    assert kept == {"cipher_system": "vigenere"}


def test_scripted_simple_substitution_rerun_without_language_or_homophonic_flags():
    """A scripted lead submits a valid simple-substitution rerun WITHOUT guessing
    `language` or unsupported homophonic flags: the config validates, the host
    stamps the run language, and the homophonic defaults are inert/untouched."""
    _ct, state = _make_state()  # state.language == "en"
    # Swap in a trivial runner for the real automated_solver type so the submit
    # completes inline without invoking the heavy solver stack (schema/defaults
    # stay the real ones).
    entry = EXPERIMENT_TYPES["automated_solver"]
    orig_runner = entry["runner"]
    entry["runner"] = lambda cipher, snapshot, config: {
        "status": "completed", "solver": "fake", "error_message": None,
        "elapsed_seconds": 0.0, "key": {}, "final_decryption": "X", "steps": [],
    }
    try:
        q = ExperimentQueue(synchronous=True)
        out = dispatch_experiment_submit(
            q, state, state.workspace, _fake_executor(),
            {
                "type": "automated_solver",
                "branch": "main",
                # No `language`, no `allow_homophones` — just the family hint.
                "config": {"cipher_system": "simple_substitution"},
            },
            1,
        )
    finally:
        entry["runner"] = orig_runner

    assert "error" not in out and "config_errors" not in out
    assert out.get("experiment_id")
    record = state.experiment_queue[-1]
    cfg = record["config"]
    # Host-derived language was stamped (the model never supplied it).
    assert cfg["language"] == "en"
    assert cfg["cipher_system"] == "simple_substitution"
    # Homophonic defaults are present but were not guessed by the model.
    assert cfg["homophonic_solver"] == "zenith_native"
    assert cfg["homophonic_refinement"] == "none"
    assert cfg["transform_search"] == "off"


# ---------------------------------------------------------------------------
# Purity / isolation (Part 7)
# ---------------------------------------------------------------------------
def test_source_branch_metadata_pipeline_byte_identical_after_run():
    _ct, state = _make_state()
    ws = state.workspace
    branch = ws.get_branch("main")
    branch.metadata["nested"] = {"pipeline": [{"op": "x"}], "notes": ["a", "b"]}
    branch.metadata["mode"] = "substitution"
    before = copy.deepcopy(branch.metadata)

    def mutating_runner(cipher, snapshot, config):
        # A misbehaving runner: mutate the snapshot it receives. Submit deep-copied
        # the source branch, so the LEAD's branch must stay byte-identical.
        snapshot["metadata"]["nested"]["notes"].append("CORRUPT")
        snapshot["metadata"]["mode"] = "CORRUPT"
        return {"status": "completed", "final_decryption": "X"}

    entry = {"config_schema": {"type": "object", "properties": {}},
             "config_defaults": {}, "runner": mutating_runner, "description": "m"}
    with _registered_type("mut", entry):
        dispatch_experiment_submit(
            ExperimentQueue(synchronous=True), state, ws, _fake_executor(),
            {"type": "mut", "branch": "main"}, 1,
        )
    assert branch.metadata == before  # source branch untouched


def test_automated_solver_runner_does_not_mutate_input_snapshot():
    from investigation.state import _serialize_branch
    _ct, state = _make_state()
    ws = state.workspace
    ws.get_branch("main").metadata["nested"] = {"k": [1, 2, 3]}
    snapshot = copy.deepcopy(_serialize_branch(ws, "main"))
    pristine = copy.deepcopy(snapshot)
    config = apply_config_defaults("automated_solver", {})
    config["model_variant"] = None
    config["language"] = "en"
    result = exp._automated_solver_runner(state.cipher, snapshot, config)
    # The F5 internal deep-copy means the input snapshot is byte-identical after.
    assert snapshot == pristine
    assert set(result) == {"status", "solver", "error_message",
                           "elapsed_seconds", "key", "final_decryption", "steps"}


# ---------------------------------------------------------------------------
# Dedup (A9) — variant-aware
# ---------------------------------------------------------------------------
def test_dedup_completed_returns_deduplicated_no_new_record():
    calls = {"n": 0}

    def runner(cipher, snapshot, config):
        calls["n"] += 1
        return {"status": "completed", "final_decryption": "OK"}

    entry = {"config_schema": {"type": "object", "properties": {}},
             "config_defaults": {}, "runner": runner, "description": "d"}
    _ct, state = _make_state()
    with _registered_type("fake", entry):
        q = ExperimentQueue(synchronous=True)
        first = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                           {"type": "fake", "branch": "main"}, 1)
        assert calls["n"] == 1
        second = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                            {"type": "fake", "branch": "main"}, 2)
    assert second.get("deduplicated") is True
    assert second["experiment_id"] == first["experiment_id"]
    assert calls["n"] == 1  # runner not re-invoked
    assert len(state.experiment_queue) == 1


def test_dedup_variant_aware():
    _ct, state = _make_state()
    # Two configs differing only by model_variant must NOT dedup.
    from investigation.state import _serialize_branch
    snap = copy.deepcopy(_serialize_branch(state.workspace, "main"))
    eff = exp._reconstruct_effective(state.cipher, snap)
    cfg_a = apply_config_defaults("automated_solver", {})
    cfg_a["model_variant"] = None
    cfg_a["language"] = "en"
    cfg_b = dict(cfg_a)
    cfg_b["model_variant"] = "some_variant"
    assert dedup_key("automated_solver", "en", eff, cfg_a) != dedup_key(
        "automated_solver", "en", eff, cfg_b)


# ---------------------------------------------------------------------------
# Concurrency + sync fallback (Part 7)
# ---------------------------------------------------------------------------
def _submit(q, state, exec_, gate_id, turn):
    return dispatch_experiment_submit(
        q, state, state.workspace, exec_,
        {"type": "block", "branch": "main", "config": {"gate_id": gate_id}}, turn,
    )


def test_concurrency_two_running_one_pending_then_promote(_arbiter_env_guard):
    gates = {g: threading.Event() for g in ("g1", "g2", "g3")}
    _ct, state = _make_state()
    with _registered_type("block", _blocking_type(gates)):
        q = ExperimentQueue(synchronous=False, workers=4, slots=2)  # S=2, I=2
        r1 = _submit(q, state, _fake_executor(), "g1", 1)
        r2 = _submit(q, state, _fake_executor(), "g2", 1)
        r3 = _submit(q, state, _fake_executor(), "g3", 1)
        statuses = {r["experiment_id"]: r["status"] for r in state.experiment_queue}
        assert statuses[r1["experiment_id"]] == "running"
        assert statuses[r2["experiment_id"]] == "running"
        assert statuses[r3["experiment_id"]] == "pending"
        # Release the two running; harvest on the lead thread.
        gates["g1"].set(); gates["g2"].set()
        assert q.wait_settled(r1["experiment_id"], timeout=5)
        assert q.wait_settled(r2["experiment_id"], timeout=5)
        transitioned = q.poll(state, 2)
        assert set(transitioned) == {r1["experiment_id"], r2["experiment_id"]}
        by_id = {r["experiment_id"]: r for r in state.experiment_queue}
        assert by_id[r1["experiment_id"]]["status"] == "completed"
        assert by_id[r2["experiment_id"]]["status"] == "completed"
        # The third was promoted to running by the same poll.
        assert by_id[r3["experiment_id"]]["status"] == "running"
        # Drain the third so no daemon thread lingers on an unset gate.
        gates["g3"].set()
        assert q.wait_settled(r3["experiment_id"], timeout=5)
        q.poll(state, 3)
        assert by_id[r3["experiment_id"]]["status"] == "completed"


def test_sync_fallback_identical_record_shape(_arbiter_env_guard):
    gates = {g: threading.Event() for g in ("g1", "g2", "g3")}
    for g in gates.values():
        g.set()  # sync runs inline on the lead thread, so gates must be pre-set

    # Async run.
    a_ct, a_state = _make_state()
    with _registered_type("block", _blocking_type(gates)):
        qa = ExperimentQueue(synchronous=False, workers=4, slots=2)
        ids = []
        for i, g in enumerate(("g1", "g2", "g3")):
            ids.append(_submit(qa, a_state, _fake_executor(), g, 1)["experiment_id"])
        # S=2, so the third stays pending until a poll promotes it. Drain in two
        # rounds (Events only): settle the first two, poll (promotes the third),
        # settle the third, poll again.
        qa.wait_settled(ids[0], timeout=5)
        qa.wait_settled(ids[1], timeout=5)
        qa.poll(a_state, 2)
        qa.wait_settled(ids[2], timeout=5)
        qa.poll(a_state, 3)
        async_records = {r["experiment_id"]: r for r in a_state.experiment_queue}

        # Sync run — identical script.
        s_ct, s_state = _make_state()
        qs = ExperimentQueue(synchronous=True, workers=4, slots=2)
        for g in ("g1", "g2", "g3"):
            _submit(qs, s_state, _fake_executor(), g, 1)
        sync_records = s_state.experiment_queue

    assert all(r["status"] == "completed" for r in async_records.values())
    assert all(r["status"] == "completed" for r in sync_records)
    # Identical record shapes (same key set).
    async_keys = {frozenset(r) for r in async_records.values()}
    sync_keys = {frozenset(r) for r in sync_records}
    assert async_keys == sync_keys


# ---------------------------------------------------------------------------
# Arbiter env: 0->1 apply, 1->0 restore, user-var not clobbered, F2 pins
# ---------------------------------------------------------------------------
@pytest.fixture
def _arbiter_env_guard():
    """Snapshot + restore the three arbiter override vars around a test."""
    saved = {v: os.environ.get(v) for v in exp._ARBITER_ENV_VARS}
    try:
        yield
    finally:
        for v, prior in saved.items():
            if prior is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = prior


def test_env_overrides_set_on_0to1_and_restored_on_1to0(_arbiter_env_guard):
    for v in exp._ARBITER_ENV_VARS:
        os.environ.pop(v, None)
    gates = {"g1": threading.Event()}
    _ct, state = _make_state()
    with _registered_type("block", _blocking_type(gates)):
        q = ExperimentQueue(synchronous=False, workers=4, slots=2)  # I=2
        r1 = _submit(q, state, _fake_executor(), "g1", 1)
        # 0->1 transition applied the overrides = str(I).
        for v in exp._ARBITER_ENV_VARS:
            assert os.environ.get(v) == str(q.I)
        gates["g1"].set()
        q.wait_settled(r1["experiment_id"], timeout=5)
        q.poll(state, 2)  # 1->0 restore
        for v in exp._ARBITER_ENV_VARS:
            assert v not in os.environ


def test_user_set_var_not_clobbered_provenance(_arbiter_env_guard):
    for v in exp._ARBITER_ENV_VARS:
        os.environ.pop(v, None)
    os.environ["DECIPHER_NULL_MASK_THREADS"] = "1"  # user wins
    gates = {"g1": threading.Event()}
    _ct, state = _make_state()
    with _registered_type("block", _blocking_type(gates)):
        q = ExperimentQueue(synchronous=False, workers=4, slots=2)
        out = _submit(q, state, _fake_executor(), "g1", 1)
        assert os.environ["DECIPHER_NULL_MASK_THREADS"] == "1"  # untouched
        assert os.environ["DECIPHER_HOMOPHONIC_PARALLEL_SEEDS"] == str(q.I)
        prov = out["arbiter"]["arbiter_overridden"]
        assert prov["DECIPHER_NULL_MASK_THREADS"] is False
        assert prov["DECIPHER_HOMOPHONIC_PARALLEL_SEEDS"] is True
        assert out.get("warnings")  # a warning is surfaced
        gates["g1"].set()
        q.wait_settled(out["experiment_id"], timeout=5)
        q.poll(state, 2)


def test_env_set_completes_before_thread_start(monkeypatch, _arbiter_env_guard):
    """F2 pin: the 0->1 env set completes BEFORE Thread.start()."""
    for v in exp._ARBITER_ENV_VARS:
        os.environ.pop(v, None)
    observed = {}
    real_thread = threading.Thread

    class RecordingThread(real_thread):
        def start(self):
            observed["at_start"] = os.environ.get("DECIPHER_HOMOPHONIC_PARALLEL_SEEDS")
            super().start()

    monkeypatch.setattr(exp.threading, "Thread", RecordingThread)
    gates = {"g1": threading.Event()}
    gates["g1"].set()
    _ct, state = _make_state()
    with _registered_type("block", _blocking_type(gates)):
        q = ExperimentQueue(synchronous=False, workers=4, slots=2)
        out = _submit(q, state, _fake_executor(), "g1", 1)
        assert observed["at_start"] == str(q.I)  # env was set before start()
        q.wait_settled(out["experiment_id"], timeout=5)
        q.poll(state, 2)


def test_finalize_env_restore_guarded_by_alive_orphan(_arbiter_env_guard):
    for v in exp._ARBITER_ENV_VARS:
        os.environ.pop(v, None)
    gates = {"g1": threading.Event()}  # never set -> thread stays alive
    _ct, state = _make_state()
    with _registered_type("block", _blocking_type(gates)):
        q = ExperimentQueue(synchronous=False, workers=4, slots=2)
        out = _submit(q, state, _fake_executor(), "g1", 1)
        # A live orphan: finalize must LEAVE the overrides in place + warn.
        warning = q.finalize_env_restore()
        assert warning is not None
        assert os.environ.get("DECIPHER_HOMOPHONIC_PARALLEL_SEEDS") == str(q.I)
        # Release; now no thread alive -> restore succeeds.
        gates["g1"].set()
        q.wait_settled(out["experiment_id"], timeout=5)
        assert q.finalize_env_restore() is None
        assert "DECIPHER_HOMOPHONIC_PARALLEL_SEEDS" not in os.environ


def test_drain_and_promote_in_one_poll_no_env_leak(_arbiter_env_guard):
    """F-1 regression: a single poll that harvests ALL running records THEN
    promotes a pending one must NOT re-capture priors (idempotent apply) —
    provenance stays correct and the overrides restore cleanly on full drain."""
    for v in exp._ARBITER_ENV_VARS:
        os.environ.pop(v, None)
    gates = {"ga": threading.Event(), "gb": threading.Event()}
    _ct, state = _make_state()
    with _registered_type("block", _blocking_type(gates)):
        q = ExperimentQueue(synchronous=False, workers=4, slots=1)  # S=1 -> serialize
        a = _submit(q, state, _fake_executor(), "ga", 1)  # running (0->1 apply)
        b = _submit(q, state, _fake_executor(), "gb", 1)  # pending
        # provenance after the first apply: the arbiter overrode all three.
        assert all(q.arbiter_status()["arbiter_overridden"].values())
        gates["ga"].set()
        q.wait_settled(a["experiment_id"], timeout=5)
        # ONE poll harvests A (running->0) AND promotes B (0->1 within the same
        # poll). The idempotent apply must keep the TRUE priors (absent) — not
        # re-capture the already-overridden env.
        q.poll(state, 2)
        by_id = {r["experiment_id"]: r for r in state.experiment_queue}
        assert by_id[a["experiment_id"]]["status"] == "completed"
        assert by_id[b["experiment_id"]]["status"] == "running"
        prov = q.arbiter_status()["arbiter_overridden"]
        assert all(prov.values()), "provenance flipped to user-set (F-1 leak)"
        assert not q.arbiter_status()["warnings"]
        for v in exp._ARBITER_ENV_VARS:
            assert os.environ.get(v) == str(q.I)  # still pinned while B runs
        # Full drain -> the three vars restore to their TRUE prior (absent).
        gates["gb"].set()
        q.wait_settled(b["experiment_id"], timeout=5)
        q.poll(state, 3)
    for v in exp._ARBITER_ENV_VARS:
        assert v not in os.environ, f"{v} leaked after full drain"


# ---------------------------------------------------------------------------
# Variant concurrency (F1c)
# ---------------------------------------------------------------------------
def test_variant_concurrency_each_worker_sees_own_variant(monkeypatch, _arbiter_env_guard):
    seen: dict[str, str | None] = {}
    resolved = threading.Semaphore(0)

    def patched(language, variant=None, models_dir=None):
        name = threading.current_thread().name
        seen[name] = variant
        if name != "MainThread":
            resolved.release()
        return None

    monkeypatch.setattr(model_registry, "resolve_language_model", patched)
    gate = threading.Event()

    def variant_runner(cipher, snapshot, config):
        from automated import runner as auto
        prev = auto.set_active_model_variant(config.get("model_variant"))
        try:
            auto._zenith_native_model_path("en")  # reads the thread-local variant
        finally:
            auto.set_active_model_variant(prev)
        gate.wait()
        return {"status": "completed", "final_decryption": "X"}

    entry = {"config_schema": {"type": "object", "properties": {}},
             "config_defaults": {}, "runner": variant_runner, "description": "v"}
    _ct, state = _make_state()
    with _registered_type("va", entry):
        q = ExperimentQueue(synchronous=False, workers=4, slots=2)
        r1 = dispatch_experiment_submit(
            q, state, state.workspace, _fake_executor(),
            {"type": "va", "branch": "main", "config": {"model_variant": "alpha"}}, 1)
        r2 = dispatch_experiment_submit(
            q, state, state.workspace, _fake_executor(),
            {"type": "va", "branch": "main", "config": {"model_variant": "beta"}}, 1)
        assert resolved.acquire(timeout=5)
        assert resolved.acquire(timeout=5)
        gate.set()
        q.wait_settled(r1["experiment_id"], timeout=5)
        q.wait_settled(r2["experiment_id"], timeout=5)
        q.poll(state, 2)

    worker_variants = sorted(v for n, v in seen.items() if n.startswith("experiment-"))
    assert worker_variants == ["alpha", "beta"]  # each worker saw its OWN variant
    recs = state.experiment_queue
    assert recs[0]["config"]["model_variant"] != recs[1]["config"]["model_variant"]
    assert recs[0]["dedup_key"] != recs[1]["dedup_key"]


# ---------------------------------------------------------------------------
# Orphan / resume round-trip (A9)
# ---------------------------------------------------------------------------
def _counting_block_type(gates, calls):
    def runner(cipher, snapshot, config):
        calls["n"] += 1
        gate = gates.get(config.get("gate_id"))
        if gate is not None:
            gate.wait()
        return {"status": "completed", "gate_id": config.get("gate_id"),
                "final_decryption": "OK"}
    return {
        "config_schema": {"type": "object", "properties": {"gate_id": {"type": "string"}}},
        "config_defaults": {"gate_id": None},
        "runner": runner,
        "description": "counting block",
    }


def test_orphan_resume_round_trip(_arbiter_env_guard):
    gates = {"gc": threading.Event(), "gr": threading.Event(), "gp": threading.Event()}
    gates["gc"].set()  # completed one pre-set
    calls = {"n": 0}
    _ct, state = _make_state()
    state.workspace.fork("work", from_branch="main")

    with _registered_type("block", _counting_block_type(gates, calls)):
        q = ExperimentQueue(synchronous=False, workers=4, slots=1)  # S=1
        c = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                       {"type": "block", "branch": "main",
                                        "config": {"gate_id": "gc"}}, 1)
        q.wait_settled(c["experiment_id"], timeout=5)
        q.poll(state, 1)  # completed; slot free
        r = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                       {"type": "block", "branch": "work",
                                        "config": {"gate_id": "gr"}}, 2)  # running
        p = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                       {"type": "block", "branch": "work",
                                        "config": {"gate_id": "gp"}}, 2)  # pending
        by_id = {rr["experiment_id"]: rr for rr in state.experiment_queue}
        assert by_id[c["experiment_id"]]["status"] == "completed"
        assert by_id[r["experiment_id"]]["status"] == "running"
        assert by_id[p["experiment_id"]]["status"] == "pending"
        calls_before_reload = calls["n"]

        # Serialize mid-run and reload.
        reloaded = InvestigationState.from_artifact_dict(
            json.loads(json.dumps(state.to_artifact_dict())))
        rl = {rr["experiment_id"]: rr for rr in reloaded.experiment_queue}
        assert rl[c["experiment_id"]]["status"] == "completed"  # intact
        assert rl[c["experiment_id"]]["result"]  # result survived
        assert rl[r["experiment_id"]]["status"] == "orphaned"
        assert rl[r["experiment_id"]]["orphan_reason"] == "loaded"
        assert rl[p["experiment_id"]]["status"] == "orphaned"

        # Resubmitting the completed spec dedups to the completed record (zero
        # runner invocations).
        q2 = ExperimentQueue(synchronous=True)
        dup = dispatch_experiment_submit(q2, reloaded, reloaded.workspace, _fake_executor(),
                                         {"type": "block", "branch": "main",
                                          "config": {"gate_id": "gc"}}, 3)
        assert dup.get("deduplicated") is True
        assert calls["n"] == calls_before_reload  # no new invocation

        # resubmit the orphaned id re-runs from the STORED snapshot after the
        # source branch is deleted.
        reloaded.workspace.delete("work")
        gates["gr"].set()  # sync runner needs its gate set
        again = dispatch_experiment_submit(
            q2, reloaded, reloaded.workspace, _fake_executor(),
            {"type": "block", "resubmit": r["experiment_id"]}, 4)
        assert calls["n"] == calls_before_reload + 1  # ran from snapshot
        new_rec = q2._find(reloaded, again["experiment_id"])
        assert new_rec is not None and new_rec["status"] == "completed"
        old_rec = q2._find(reloaded, r["experiment_id"])
        assert old_rec["superseded_by"] == again["experiment_id"]

        # cleanup the still-blocked running thread on the ORIGINAL queue and poll
        # it to completion so the arbiter env overrides restore (1->0).
        gates["gr"].set()
        gates["gp"].set()
        q.wait_settled(r["experiment_id"], timeout=5)
        q.poll(state, 5)  # harvests r, promotes p -> running
        q.wait_settled(p["experiment_id"], timeout=5)
        q.poll(state, 6)  # harvests p -> running 0 -> env restored


# ---------------------------------------------------------------------------
# run_v3-driven tests (scripted lead + finalize orphaning)
# ---------------------------------------------------------------------------
from agent.model_provider import ModelResponse, ModelUsage, TextBlock, ToolUseBlock
from investigation import sessions as sessions_mod
from investigation.loop_v3 import run_v3
from investigation.sessions import SessionCapabilities
from investigation.state import BudgetEntry


def _resp(content):
    return ModelResponse(content=content, usage=ModelUsage(100, 10, 0))


class _VerifyFake:
    """M5: a verify episode worker (empty toolset -> zero-tool submit)."""

    def __init__(self, provider=None, system="", role="episode:verify"):
        self.model = "fake-luna"
        self.provider_name = "openai"
        self.capabilities = SessionCapabilities()
        self._budget = []

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("episode:verify", "openai", "fake-luna", 40, 8, 0))
        return _resp([ToolUseBlock(id="v1", name="episode_submit_result",
                      input={"result": {"coherence": 8, "reader_accepts": True,
                                        "reader_accepts_as_solution": True,
                                        "gloss": "reads", "anomalies": [],
                                        "confidence": "high"},
                             "summary": "reads well"})])

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": "openai", "model": self.model, "exchanges": []}


def _experiment_ids_from_blocks(blocks):
    """Accumulate submit-result experiment ids (submit responses carry `arbiter`)."""
    ids = []
    for m in blocks:
        for b in (m.get("content") or []):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                try:
                    data = json.loads(b.get("content") or "")
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(data, dict) and "arbiter" in data and data.get("experiment_id"):
                    ids.append(data["experiment_id"])
    return ids


class FlowLead:
    def __init__(self, queue, gates):
        self.model = "fake-lead"; self.provider_name = "openai"
        self.capabilities = SessionCapabilities(); self._budget = []
        self._step = 0; self.blocks_seen = []; self._queue = queue
        self._gates = gates; self._ids = []

    def send(self, blocks, tools=None, max_tokens=8192):
        self.blocks_seen.append(json.dumps(blocks, default=str))
        self._budget.append(BudgetEntry("lead", "openai", "fake-lead", 100, 10, 0))
        for eid in _experiment_ids_from_blocks(blocks):
            if eid not in self._ids:
                self._ids.append(eid)
        step = self._step; self._step += 1
        if step == 0:
            return _resp([ToolUseBlock(id="s1", name="experiment_submit",
                          input={"type": "block", "branch": "main", "config": {"gate_id": "ga"}})])
        if step == 1:
            return _resp([ToolUseBlock(id="s2", name="experiment_submit",
                          input={"type": "block", "branch": "main", "config": {"gate_id": "gb"}})])
        if step in (2, 3):
            return _resp([ToolUseBlock(id=f"w{step}", name="decode_show", input={"branch": "main"})])
        if step == 4:
            self._gates["ga"].set(); self._gates["gb"].set()
            for eid in self._ids:
                self._queue.wait_settled(eid, timeout=5)
            return _resp([ToolUseBlock(id="c1", name="experiment_collect",
                          input={"experiment_id": self._ids[0], "install": True})])
        if step == 5:
            return _resp([ToolUseBlock(id="c2", name="experiment_collect",
                          input={"experiment_id": self._ids[1]})])
        if step == 6:
            # M5: verify `main` before declaring it.
            return _resp([ToolUseBlock(id="ev", name="episode_run",
                          input={"kind": "verify", "goal": "verify main",
                                 "branches": ["main"]})])
        return _resp([TextBlock(text="done"),
                      ToolUseBlock(id="d1", name="meta_declare_solution",
                                   input={"branch": "main", "rationale": "reads",
                                          "self_confidence": 0.9})])

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": "openai", "model": self.model, "exchanges": []}


def test_scripted_lead_flow(_arbiter_env_guard):
    gates = {"ga": threading.Event(), "gb": threading.Event()}
    calls = {"n": 0}
    ct, _state = _make_state()
    sessions_mod.register_session_builder("episode:verify", _VerifyFake)
    try:
        with _registered_type("block", _counting_block_type(gates, calls)):
            queue = ExperimentQueue(synchronous=False, workers=4, slots=2)
            lead = FlowLead(queue, gates)
            art = run_v3(ct, session=lead, language="en", max_iterations=9,
                         cipher_id="v3_exp_flow", experiment_queue=queue)
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)

    # experiment_complete evidence entries (both experiments).
    completes = [e for e in art.investigation_state["evidence_log"]
                 if e["kind"] == "experiment_complete"]
    assert len(completes) == 2

    # The queue section rendered BOTH experiments while running (work turns).
    work_blocks = lead.blocks_seen[2] + lead.blocks_seen[3]
    assert "Experiment queue" in work_blocks
    assert "[running]" in work_blocks

    # experiment:<fake> bucket present with calls == 2.
    bucket = art.budget_by_category.get("experiment:block")
    assert bucket is not None and bucket["calls"] == 2
    assert bucket["input_tokens"] == 0 and bucket["cost_usd"] == 0.0

    # One experiment installed as a fresh branch.
    installed = [r for r in art.experiments if r.get("installed_as")]
    assert len(installed) == 1
    assert any(b.name == installed[0]["installed_as"] for b in art.branches)

    # artifact.experiments carries both records; both collected + completed.
    assert len(art.experiments) == 2
    assert all(r["status"] == "completed" for r in art.experiments)
    assert all(r["collected"] for r in art.experiments)
    assert art.status == "solved"


class _SubmitDeclareLead:
    """t1 submit a blocked experiment; t2 declare (run_ended orphaning)."""

    def __init__(self):
        self.model = "fake-lead"; self.provider_name = "openai"
        self.capabilities = SessionCapabilities(); self._budget = []; self._step = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("lead", "openai", "fake-lead", 100, 10, 0))
        step = self._step; self._step += 1
        if step == 0:
            return _resp([ToolUseBlock(id="s1", name="experiment_submit",
                          input={"type": "block", "branch": "main", "config": {"gate_id": "gx"}})])
        return _resp([ToolUseBlock(id="d1", name="meta_declare_solution",
                      input={"branch": "main", "rationale": "done", "self_confidence": 0.9})])

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": "openai", "model": self.model, "exchanges": []}


def test_finalize_orphaning_run_ended(_arbiter_env_guard):
    gates = {"gx": threading.Event()}  # never set during the run
    calls = {"n": 0}
    ct, _state = _make_state()
    with _registered_type("block", _counting_block_type(gates, calls)):
        queue = ExperimentQueue(synchronous=False, workers=4, slots=2)
        art = run_v3(ct, session=_SubmitDeclareLead(), language="en", max_iterations=5,
                     cipher_id="v3_run_ended", experiment_queue=queue)
        assert len(art.experiments) == 1
        assert art.experiments[0]["status"] == "orphaned"
        assert art.experiments[0]["orphan_reason"] == "run_ended"
        # A live orphan thread -> the env override was retained + warned.
        assert any(e.event == "experiment_env_override_retained" for e in art.loop_events)
        assert os.environ.get("DECIPHER_HOMOPHONIC_PARALLEL_SEEDS") == str(queue.I)
        # cleanup.
        gates["gx"].set()
        queue.wait_settled(art.experiments[0]["experiment_id"], timeout=5)
        queue.finalize_env_restore()


def test_finalize_orphaning_stopped(monkeypatch, _arbiter_env_guard):
    from agent import tools_v2
    gates = {"gx": threading.Event()}  # unset -> the experiment is still running
    calls = {"n": 0}

    def boom(self, name, args, tool_use_id=None):
        raise KeyboardInterrupt

    ct, _state = _make_state()
    with _registered_type("block", _counting_block_type(gates, calls)):
        queue = ExperimentQueue(synchronous=False, workers=4, slots=2)
        # experiment_submit is routed before executor.execute; the interrupt fires
        # on the t2 decode_show.
        monkeypatch.setattr(tools_v2.WorkspaceToolExecutor, "execute", boom)
        art = run_v3(ct, session=_InterruptLead(), language="en", max_iterations=5,
                     cipher_id="v3_stopped", experiment_queue=queue)
    assert art.status == "stopped"
    assert len(art.experiments) == 1
    assert art.experiments[0]["status"] == "orphaned"
    assert art.experiments[0]["orphan_reason"] == "interrupted"
    # cleanup the still-blocked orphan thread + retained env.
    gates["gx"].set()
    queue.wait_settled(art.experiments[0]["experiment_id"], timeout=5)
    queue.finalize_env_restore()


class _InterruptLead:
    def __init__(self):
        self.model = "fake-lead"; self.provider_name = "openai"
        self.capabilities = SessionCapabilities(); self._budget = []; self._step = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("lead", "openai", "fake-lead", 100, 10, 0))
        step = self._step; self._step += 1
        if step == 0:
            return _resp([ToolUseBlock(id="s1", name="experiment_submit",
                          input={"type": "block", "branch": "main", "config": {"gate_id": "gx"}})])
        return _resp([ToolUseBlock(id="w1", name="decode_show", input={"branch": "main"})])

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": "openai", "model": self.model, "exchanges": []}


# ---------------------------------------------------------------------------
# Null-mask collect (F7) + packet-stripping pin
# ---------------------------------------------------------------------------
def _real_executor(state):
    from analysis import dictionary, pattern
    dict_path = dictionary.get_dictionary_path("en")
    word_set = dictionary.load_word_set(dict_path) if dict_path else set()
    word_list = pattern.load_word_list(dict_path) if dict_path else []
    pattern_dict = pattern.build_pattern_dictionary(word_list)
    return WorkspaceToolExecutor(
        workspace=state.workspace, language="en", word_set=word_set,
        word_list=word_list, pattern_dict=pattern_dict,
        declaration_policy=NoGatesPolicy(),
        finalist_sessions=state.finalist_sessions,
    )


def _null_mask_result_type():
    null_step = {
        "name": "search_null_masks",
        "status": "completed",
        "selected": {"decryption": "SELECTED READING", "mask": [1],
                     "validation_score_v2": 0.5, "selection_score": 0.4},
        "top_finalists": [
            {"mask": [1], "status": "completed", "decryption": "READING ONE",
             "validation_score_v2": 0.5, "selection_score": 0.4, "candidate_id": "m1"},
            {"mask": [2], "status": "completed", "decryption": "READING TWO",
             "validation_score_v2": 0.4, "selection_score": 0.3, "candidate_id": "m2"},
        ],
        "evaluated_rows": [], "candidate_symbols": [],
        # A runner-attached packet that must NEVER reach the model.
        "packet": {"secret": "artifact-only"},
    }
    route_step = {"name": "route_automated_solver", "packet": {"x": 1}}

    def runner(cipher, snapshot, config):
        return {
            "status": "completed", "solver": "zenith_native",
            "final_decryption": "SELECTED READING", "error_message": "",
            "key": {"0": 0, "1": 1}, "steps": [route_step, null_step],
        }
    return {"config_schema": {"type": "object", "properties": {}},
            "config_defaults": {}, "runner": runner, "description": "nm"}


def test_collect_null_mask_session_and_packet_stripping():
    _ct, state = _make_state()
    executor = _real_executor(state)
    with _registered_type("nm", _null_mask_result_type()):
        q = ExperimentQueue(synchronous=True)
        sub = dispatch_experiment_submit(q, state, state.workspace, executor,
                                         {"type": "nm", "branch": "main"}, 1)
        packet = dispatch_experiment_collect(
            q, state, state.workspace, executor,
            {"experiment_id": sub["experiment_id"], "install": True}, 2)

    # Null-mask session surfaced (F7): id + review + suggested triplet.
    assert packet["null_mask_search_session_id"]
    review = packet["null_mask_finalist_review"]
    assert review["finalist_review_count"] >= 1
    assert packet["suggested_next_tools"] == [
        "search_review_null_mask_finalists",
        "act_rate_null_mask_finalist",
        "act_install_null_mask_finalists",
    ]
    # Installed branch exists; source branch still live -> session source = it.
    installed = packet["installed_as"]
    assert state.workspace.has_branch(installed)
    session = state.finalist_sessions.get("null_mask", packet["null_mask_search_session_id"])
    assert session["source_branch"] == "main"

    # Packet-stripping pin: the route/primary steps carry no "packet" key.
    assert "packet" not in json.dumps(packet["route_step"])
    assert "packet" not in json.dumps(packet["primary_step"])


# ---------------------------------------------------------------------------
# Pins (Part 7)
# ---------------------------------------------------------------------------
def test_pin_tool_definitions_length_unchanged():
    from agent.tools_v2 import TOOL_DEFINITIONS, VALID_TOOL_NAMES
    names = {d["name"] for d in TOOL_DEFINITIONS}
    assert "experiment_submit" not in names
    assert "experiment_collect" not in names
    assert "experiment_submit" not in VALID_TOOL_NAMES


def test_pin_executor_execute_experiment_submit_is_unknown_tool():
    _ct, state = _make_state()
    executor = _real_executor(state)
    result = executor.execute("experiment_submit", {}, tool_use_id="t1")
    assert "error" in json.loads(result)


def test_pin_episode_toolset_rejects_experiment_tools():
    from investigation.episodes import _validate_episode_toolset
    with pytest.raises(ValueError):
        _validate_episode_toolset({"experiment_submit"})
    with pytest.raises(ValueError):
        _validate_episode_toolset({"experiment_collect"})


# ---------------------------------------------------------------------------
# Dedup: already_queued + orphaned -> new record + superseded_by (A9)
# ---------------------------------------------------------------------------
def test_dedup_already_queued_when_running(_arbiter_env_guard):
    gates = {"g1": threading.Event()}
    _ct, state = _make_state()
    with _registered_type("block", _blocking_type(gates)):
        q = ExperimentQueue(synchronous=False, workers=4, slots=2)
        first = _submit(q, state, _fake_executor(), "g1", 1)
        again = _submit(q, state, _fake_executor(), "g1", 1)  # same spec while running
        assert again.get("already_queued") is True
        assert again["experiment_id"] == first["experiment_id"]
        assert len(state.experiment_queue) == 1
        gates["g1"].set()
        q.wait_settled(first["experiment_id"], timeout=5)
        q.poll(state, 2)


def test_dedup_orphaned_creates_new_and_stamps_superseded():
    _ct, state = _make_state()
    # Hand-craft an orphaned record with a known dedup key matching a fresh submit.
    from investigation.state import _serialize_branch
    snap = copy.deepcopy(_serialize_branch(state.workspace, "main"))
    eff = exp._reconstruct_effective(state.cipher, snap)
    cfg = apply_config_defaults("block", {"gate_id": "g9"})
    cfg["model_variant"] = None
    cfg["language"] = "en"
    key = dedup_key("block", "en", eff, cfg)
    orphan = {"experiment_id": "old12", "dedup_key": key, "type": "block",
              "branch": "main", "snapshot": snap, "config": cfg, "status": "orphaned",
              "started_at": 1.0, "collected": False, "superseded_by": None}
    state.experiment_queue.append(orphan)
    gates = {"g9": threading.Event()}; gates["g9"].set()
    with _registered_type("block", _blocking_type(gates)):
        q = ExperimentQueue(synchronous=True)
        out = _submit(q, state, _fake_executor(), "g9", 5)
    assert out["experiment_id"] != "old12"  # new record
    assert orphan["superseded_by"] == out["experiment_id"]
    assert len(state.experiment_queue) == 2


def test_resubmit_running_id_returns_already_queued_not_duplicate(_arbiter_env_guard):
    """F-2: resubmit of a still-running experiment must not duplicate live work."""
    gates = {"g1": threading.Event()}
    _ct, state = _make_state()
    with _registered_type("block", _blocking_type(gates)):
        q = ExperimentQueue(synchronous=False, workers=4, slots=2)
        first = _submit(q, state, _fake_executor(), "g1", 1)  # running
        out = dispatch_experiment_submit(
            q, state, state.workspace, _fake_executor(),
            {"type": "block", "resubmit": first["experiment_id"]}, 2)
        assert out.get("already_queued") is True
        assert out["experiment_id"] == first["experiment_id"]
        assert len(state.experiment_queue) == 1  # no duplicate
        assert state.experiment_queue[0]["superseded_by"] is None  # not stamped
        gates["g1"].set()
        q.wait_settled(first["experiment_id"], timeout=5)
        q.poll(state, 3)


def test_resubmit_completed_source_dedups(_arbiter_env_guard):
    """F-2: resubmit of a completed source returns deduplicated (not an error)."""
    gates = {"g1": threading.Event()}; gates["g1"].set()
    _ct, state = _make_state()
    with _registered_type("block", _blocking_type(gates)):
        q = ExperimentQueue(synchronous=True)
        first = _submit(q, state, _fake_executor(), "g1", 1)  # completes inline
        out = dispatch_experiment_submit(
            q, state, state.workspace, _fake_executor(),
            {"type": "block", "resubmit": first["experiment_id"]}, 2)
    assert out.get("deduplicated") is True
    assert out["experiment_id"] == first["experiment_id"]


def test_collect_no_id_lists_and_summarizes():
    gates = {"g1": threading.Event()}; gates["g1"].set()
    _ct, state = _make_state()
    with _registered_type("block", _blocking_type(gates)):
        q = ExperimentQueue(synchronous=False, workers=4, slots=2)
        sub = _submit(q, state, _fake_executor(), "g1", 1)
        q.wait_settled(sub["experiment_id"], timeout=5)
        out = dispatch_experiment_collect(q, state, state.workspace, _fake_executor(),
                                          {}, 2)
    assert out["queue"] and any(sub["experiment_id"] in line for line in out["queue"])
    assert out["newly_completed"] and out["newly_completed"][0]["status"] == "completed"
    assert "slots" in out


# ---------------------------------------------------------------------------
# quagmire3_shotgun experiment type (spec: experiment_quagmire3_shotgun_spec.md)
# ---------------------------------------------------------------------------
import analysis.polyalphabetic_fast as _pf


def _q3_candidates():
    """Two engine-shaped candidates; each carries the fields the installer reads
    plus a bulky internal that the runner MUST drop."""
    return [
        {
            "score": -3.1, "selection_score": 0.9, "period": 8,
            "plaintext": "THE QUICK BROWN FOX", "preview": "THE QUICK BROWN FOX",
            "key": "CYCLEA", "shifts": [0, 1, 2],
            "metadata": {
                "alphabet_keyword": "SECRET", "cycleword": "CYCLEA",
                "cycleword_shifts": [0, 1, 2], "plaintext_alphabet": "ABCDE",
                "ciphertext_alphabet": "XYZWV", "quagmire_type": "quag3",
                "start_type": "keyword", "bulky_internal": "DROP ME",
            },
        },
        {
            "score": -5.0, "selection_score": 0.5, "period": 8,
            "plaintext": "SECOND CANDIDATE TEXT", "preview": "SECOND CANDIDATE TEXT",
            "key": "CYCLEB", "shifts": [1, 2, 3],
            "metadata": {
                "alphabet_keyword": "OTHER", "cycleword": "CYCLEB",
                "quagmire_type": "quag3", "start_type": "random",
            },
        },
    ]


def _fake_q3_engine(*, captured=None, candidates=None, status="completed", error=None):
    cands = _q3_candidates() if candidates is None else candidates

    def engine(cipher_text, *, language="en", keyword_lengths=None,
               cycleword_lengths=None, hillclimbs=500, restarts=1000, seed=1,
               top_n=10, threads=0, **extra):
        if captured is not None:
            captured.update({
                "language": language, "keyword_lengths": keyword_lengths,
                "cycleword_lengths": cycleword_lengths, "hillclimbs": hillclimbs,
                "restarts": restarts, "seed": seed, "top_n": top_n,
                "threads": threads,
            })
        out = {"status": status, "top_candidates": [dict(c) for c in cands]}
        if error is not None:
            out["error"] = error
        return out

    return engine


def _patch_q3_engine(monkeypatch, engine):
    monkeypatch.setattr(_pf, "search_quagmire3_shotgun_fast", engine)


def test_q3_registry_and_schema_surface():
    assert "quagmire3_shotgun" in EXPERIMENT_TYPES
    schema = model_facing_config_schema("quagmire3_shotgun")
    props = schema["properties"]
    assert set(props) == {
        "keyword_lengths", "cycleword_lengths", "hillclimbs", "restarts",
        "model_variant",
    }
    assert props["hillclimbs"]["default"] == 5000
    assert props["restarts"]["default"] == 250
    assert props["keyword_lengths"]["default"] == [7]
    assert props["cycleword_lengths"]["default"] == [8]
    assert all(props[k].get("description") for k in props)
    # Submit-tool config is the anyOf union carrying both titles.
    cfg = EXPERIMENT_SUBMIT_TOOL["input_schema"]["properties"]["config"]
    titles = {b.get("title") for b in cfg["anyOf"]}
    assert titles == {"automated_solver config", "quagmire3_shotgun config"}
    q3_branch = next(b for b in cfg["anyOf"] if b["title"] == "quagmire3_shotgun config")
    assert {k: v for k, v in q3_branch.items() if k != "title"} == schema


def test_q3_misroute_guard_redirects_to_shotgun():
    for hint in ("quagmire3", "Quagmire III"):
        errors = validate_experiment_config("automated_solver", {"cipher_system": hint})
        assert errors and any("quagmire3_shotgun" in e for e in errors)
    # The corrected example is genuinely valid and drops the quag hint.
    fixed = corrected_config_example("automated_solver", {"cipher_system": "quagmire3"})
    assert validate_experiment_config("automated_solver", fixed) == []
    assert "cipher_system" not in fixed
    # A non-quag hint still validates.
    assert validate_experiment_config("automated_solver", {"cipher_system": "vigenere"}) == []


def test_q3_runner_shape_and_purity(monkeypatch):
    from investigation.state import _serialize_branch
    _patch_q3_engine(monkeypatch, _fake_q3_engine())
    _ct, state = _make_state()
    snapshot = copy.deepcopy(_serialize_branch(state.workspace, "main"))
    pristine = copy.deepcopy(snapshot)
    config = apply_config_defaults("quagmire3_shotgun", {})
    config["model_variant"] = None
    config["language"] = "en"
    result = exp._quagmire3_shotgun_runner(state.cipher, snapshot, config)
    assert snapshot == pristine  # input snapshot byte-identical
    assert result["key"] == {}
    assert result["solver"] == "quagmire3_shotgun_rust"
    assert result["final_decryption"] == "THE QUICK BROWN FOX"
    assert result["nominal_proposals"] == 1 * 1 * 250 * 5000
    step = result["steps"][0]
    assert step["name"] == "search_quagmire3_shotgun"
    assert step["engine"] == "rust_shotgun"
    assert step["candidates"] == 2
    # Bulky engine internals dropped from the stored candidate metadata.
    assert "bulky_internal" not in result["top_candidates"][0]["metadata"]


def test_q3_runner_clamps_config(monkeypatch):
    captured: dict = {}
    _patch_q3_engine(monkeypatch, _fake_q3_engine(captured=captured))
    _ct, state = _make_state()
    from investigation.state import _serialize_branch
    snapshot = copy.deepcopy(_serialize_branch(state.workspace, "main"))
    config = {
        "hillclimbs": 10 ** 9, "restarts": 0,
        "keyword_lengths": [1, 25, 7], "cycleword_lengths": [],
        "language": "en",
    }
    exp._quagmire3_shotgun_runner(state.cipher, snapshot, config)
    assert captured["hillclimbs"] == 50_000
    assert captured["restarts"] == 1
    assert captured["keyword_lengths"] == [7]   # out-of-range filtered
    assert captured["cycleword_lengths"] == [8]  # emptied -> default
    assert captured["threads"] == 0


def test_q3_end_to_end_install(monkeypatch):
    from agent.loop_shared import _decoded_text_for_panel
    _patch_q3_engine(monkeypatch, _fake_q3_engine())
    _ct, state = _make_state()
    q = ExperimentQueue(synchronous=True)
    sub = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                     {"type": "quagmire3_shotgun", "branch": "main",
                                      "config": {}}, 1)
    assert sub.get("experiment_id")
    packet = dispatch_experiment_collect(
        q, state, state.workspace, _fake_executor(),
        {"experiment_id": sub["experiment_id"], "install": True}, 2)
    name = packet["installed_as"]
    branch = state.workspace.get_branch(name)
    assert branch.metadata["cipher_mode"] == "quagmire3"
    assert branch.metadata["decoded_text"] == "THE QUICK BROWN FOX"
    assert {"hypothesis", "mode:quagmire3", "mode:keyed_tableau_polyalphabetic"} <= set(branch.tags)
    # Ranked candidates summarized in the packet (previews, not full plaintext).
    assert [c["rank"] for c in packet["candidates"]] == [1, 2]
    # Gate-reachability: the panel text equals the installed candidate plaintext.
    assert _decoded_text_for_panel(state.workspace, name) == "THE QUICK BROWN FOX"


def test_q3_install_strips_inherited_null_mask_shadow(monkeypatch):
    """A null-mask block inherited from the source snapshot must not shadow the
    quagmire decoded_text in _metadata_decoded_text (review finding 1)."""
    from agent.loop_shared import _decoded_text_for_panel
    _patch_q3_engine(monkeypatch, _fake_q3_engine())
    _ct, state = _make_state()
    src = state.workspace.get_branch("main")
    src.metadata["null_mask_selected"] = {"mask": ["Z"], "selection_score": 0.5}
    q = ExperimentQueue(synchronous=True)
    sub = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                     {"type": "quagmire3_shotgun", "branch": "main",
                                      "config": {}}, 1)
    packet = dispatch_experiment_collect(
        q, state, state.workspace, _fake_executor(),
        {"experiment_id": sub["experiment_id"], "install": True}, 2)
    name = packet["installed_as"]
    branch = state.workspace.get_branch(name)
    assert "null_mask_selected" not in branch.metadata
    assert "null_mask_finalist" not in branch.metadata
    assert _decoded_text_for_panel(state.workspace, name) == "THE QUICK BROWN FOX"
    # Source branch untouched.
    assert "null_mask_selected" in src.metadata


def test_q3_install_rejects_empty_plaintext_candidate(monkeypatch):
    """An empty-plaintext candidate must error instead of installing a branch
    whose panel text falls back to the stale inherited key (review finding 2)."""
    cands = _q3_candidates()
    cands[0]["plaintext"] = ""
    _patch_q3_engine(monkeypatch, _fake_q3_engine(candidates=cands))
    _ct, state = _make_state()
    before = set(state.workspace.branch_names())
    q = ExperimentQueue(synchronous=True)
    sub = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                     {"type": "quagmire3_shotgun", "branch": "main",
                                      "config": {}}, 1)
    out = dispatch_experiment_collect(
        q, state, state.workspace, _fake_executor(),
        {"experiment_id": sub["experiment_id"], "install": True}, 2)
    assert "no plaintext" in out["error"]
    assert set(state.workspace.branch_names()) == before
    # Rank 2 (non-empty plaintext) still installs from the same record.
    out2 = dispatch_experiment_collect(
        q, state, state.workspace, _fake_executor(),
        {"experiment_id": sub["experiment_id"], "install": True,
         "candidate_rank": 2}, 3)
    branch = state.workspace.get_branch(out2["installed_as"])
    assert branch.metadata["decoded_text"] == "SECOND CANDIDATE TEXT"


def test_q3_candidate_rank_selection_and_errors(monkeypatch):
    _patch_q3_engine(monkeypatch, _fake_q3_engine())
    _ct, state = _make_state()
    q = ExperimentQueue(synchronous=True)
    sub = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                     {"type": "quagmire3_shotgun", "branch": "main",
                                      "config": {}}, 1)
    eid = sub["experiment_id"]
    # rank 2 installs candidate 2.
    p2 = dispatch_experiment_collect(
        q, state, state.workspace, _fake_executor(),
        {"experiment_id": eid, "install": True, "candidate_rank": 2}, 2)
    assert state.workspace.get_branch(p2["installed_as"]).metadata["decoded_text"] == "SECOND CANDIDATE TEXT"
    # rank 99 -> structured error naming the valid range.
    bad = dispatch_experiment_collect(
        q, state, state.workspace, _fake_executor(),
        {"experiment_id": eid, "install": True, "candidate_rank": 99}, 3)
    assert "out of range" in bad["error"]


def test_q3_candidate_rank_on_automated_solver_is_error():
    _ct, state = _make_state()
    entry = EXPERIMENT_TYPES["automated_solver"]
    orig = entry["runner"]
    entry["runner"] = lambda cipher, snapshot, config: {
        "status": "completed", "solver": "fake", "error_message": None,
        "elapsed_seconds": 0.0, "key": {}, "final_decryption": "X", "steps": [],
    }
    try:
        q = ExperimentQueue(synchronous=True)
        sub = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                         {"type": "automated_solver", "branch": "main",
                                          "config": {}}, 1)
        out = dispatch_experiment_collect(
            q, state, state.workspace, _fake_executor(),
            {"experiment_id": sub["experiment_id"], "candidate_rank": 2}, 2)
    finally:
        entry["runner"] = orig
    assert "candidate_rank is only supported" in out["error"]


def test_q3_engine_unavailable_fails_loudly_and_resubmits(monkeypatch):
    def boom(*a, **k):
        raise ImportError("fast kernel not built")

    _patch_q3_engine(monkeypatch, boom)
    _ct, state = _make_state()
    q = ExperimentQueue(synchronous=True)
    sub = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                     {"type": "quagmire3_shotgun", "branch": "main",
                                      "config": {}}, 1)
    rec = q._find(state, sub["experiment_id"])
    assert rec["status"] == "failed"
    assert "build_rust_fast.sh" in (rec["error"] or "")
    # Resubmit creates a fresh record from the stored snapshot.
    again = dispatch_experiment_submit(
        q, state, state.workspace, _fake_executor(),
        {"type": "quagmire3_shotgun", "resubmit": sub["experiment_id"]}, 2)
    assert again["experiment_id"] != sub["experiment_id"]
    assert rec["superseded_by"] == again["experiment_id"]


def test_q3_no_candidates_install_errors_no_branch(monkeypatch):
    _patch_q3_engine(monkeypatch, _fake_q3_engine(candidates=[]))
    _ct, state = _make_state()
    q = ExperimentQueue(synchronous=True)
    before = set(state.workspace.branch_names())
    sub = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                     {"type": "quagmire3_shotgun", "branch": "main",
                                      "config": {}}, 1)
    out = dispatch_experiment_collect(
        q, state, state.workspace, _fake_executor(),
        {"experiment_id": sub["experiment_id"], "install": True}, 2)
    assert "no quagmire3 candidates" in out["error"]
    assert set(state.workspace.branch_names()) == before  # no branch created


def test_q3_dedup_config_aware(monkeypatch):
    _patch_q3_engine(monkeypatch, _fake_q3_engine())
    _ct, state = _make_state()
    q = ExperimentQueue(synchronous=True)
    first = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                       {"type": "quagmire3_shotgun", "branch": "main",
                                        "config": {}}, 1)
    dup = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                     {"type": "quagmire3_shotgun", "branch": "main",
                                      "config": {}}, 2)
    assert dup.get("deduplicated") is True
    assert dup["experiment_id"] == first["experiment_id"]
    # A different hillclimbs budget is a distinct experiment.
    other = dispatch_experiment_submit(q, state, state.workspace, _fake_executor(),
                                       {"type": "quagmire3_shotgun", "branch": "main",
                                        "config": {"hillclimbs": 1234}}, 3)
    assert other["experiment_id"] != first["experiment_id"]
    assert not other.get("deduplicated")
