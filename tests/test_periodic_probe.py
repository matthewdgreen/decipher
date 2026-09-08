from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from automated import periodic_probe as probe
from automated import bounded_process as bounded
from automated import runner
from models.alphabet import Alphabet
from models.cipher_text import CipherText


def cipher(text="A" * 200):
    return CipherText(raw=text, alphabet=Alphabet.from_text(text), separator=None)


def validation(**overrides):
    return {"validation_label": "coherent_candidate", "validation_score": 2.,
            "quadgram_loglik_per_gram": -3., "strict_word_hit_score": .4,
            "segmentation": {"dict_rate": .9, "pseudo_word_fraction": .1}, **overrides}


def candidate(**overrides):
    return {"plaintext": "A" * 200, "variant": "vigenere", "period": 4,
            "shifts": [0] * 4, "key": "AAAA", **overrides}


@pytest.fixture(autouse=True)
def clean_knobs(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("DECIPHER_PERIODIC_", "DECIPHER_POLYALPHABETIC_",
                           "DECIPHER_QUAGMIRE_", "DECIPHER_KEYED_VIGENERE_")):
            monkeypatch.delenv(key)


def test_period_gates_harmonics_ties_and_bounds():
    # Deliberately controlled statistics, not inspection of R5 fresh cases.
    table = {p: .04 for p in range(2, 21)}
    table.update({4: .068, 8: .071, 12: .069, 16: .073, 20: .065})
    result = probe.select_periods(table, .041, 700, {4: 20})
    assert result["periods"][0] == 4
    assert len(result["periods"]) == 2
    assert max(probe.select_periods(table, .041, 200, {})["periodic_ic"]) == 8
    assert probe.select_periods(table, .075, 700, {})["periods"] == []
    assert probe.select_periods({p: .04 for p in range(2, 10)}, .039, 300, {})["periods"] == []
    assert probe.select_periods({}, .04, 200, {})["periods"] == []


@pytest.mark.parametrize("options,reason", [
    ({"language": "de"}, "language_out_of_scope"),
    ({"cipher_system": "vigenere"}, "explicit_direction_preserved"),
    ({"solver_hints": {"known_params": {}}}, "explicit_direction_preserved"),
    ({"transform_pipeline": {}}, "explicit_direction_preserved"),
    ({"transform_search": "auto"}, "explicit_direction_preserved"),
    ({"model_variant": "zenith_upstream"}, "custom_search_configuration_preserved"),
])
def test_directed_inputs_do_not_probe(options, reason):
    args = {"language": "en", **options}
    assert probe.diagnose(cipher(), **args)["reason"] == reason


def test_size_symbols_and_custom_modes(monkeypatch):
    for length in (179, 1201):
        assert probe.diagnose(cipher("A" * length), "en")["reason"] == "length_out_of_scope"
    assert probe.diagnose(cipher("1" * 200), "en")["reason"] == "non_a_z_symbols"
    monkeypatch.setenv("DECIPHER_KEYED_VIGENERE_MODE", "replay")
    assert probe.diagnose(cipher(), "en")["reason"] == "custom_search_configuration_preserved"
    monkeypatch.setenv("DECIPHER_PERIODIC_ROUTING", "unknown")
    with pytest.raises(ValueError, match="must be off"):
        runner.run_automated(cipher())


def test_strict_replay_periodic_and_quagmire():
    assert probe.replay_matches("A" * 200, candidate(engine="periodic"))
    for changes in ({"plaintext": "A" * 199}, {"shifts": [0]}, {"shifts": [False] * 4},
                    {"variant": "unknown"}, {"plaintext": " " * 200}):
        assert not probe.replay_matches("A" * 200, candidate(engine="periodic", **changes))
    row = candidate(engine="quagmire3", metadata={"alphabet_keyword": "ABCDEF", "cycleword": "AAAA"})
    assert probe.replay_matches("A" * 200, row)
    row["metadata"]["cycleword"] = "AAA"
    assert not probe.replay_matches("A" * 200, row)


@pytest.mark.parametrize("bad", [
    {"validation_label": "plausible_candidate"}, {"validation_score": float("nan")},
    {"strict_word_hit_score": .24}, {"segmentation": {"dict_rate": .77, "pseudo_word_fraction": .1}},
    {"segmentation": {"dict_rate": .9, "pseudo_word_fraction": .26}},
])
def test_gate_requires_every_condition(bad):
    assert probe.language_gate(validation())
    assert not probe.language_gate(validation(**bad))


def test_search_stages_budget_and_mode_state():
    events, calls = [], []
    def ordinary(*args, **kwargs):
        calls.append(kwargs)
        return {"solver": "periodic_polyalphabetic_screen", "top_candidates": [candidate()]}
    def never(*args, **kwargs):
        raise AssertionError("strong ordinary candidate must short circuit Quagmire")
    request = {"ciphertext": "A" * 200, "language": "en", "periods": [4]}
    probe.probe_search(request, events.append, periodic=ordinary, quagmire=never,
                       validator=lambda *a, **k: validation())
    assert calls[0]["periods"] == [4] and calls[0]["refine"]
    assert calls[0]["variants"] == list(probe.VARIANTS)
    selected = events[-1]["candidates"][0]
    assert selected["delivery_eligible"] and selected["replay_consistent"]
    assert probe.selected_step(selected)["key_type"] == "PeriodicShiftKey"
    assert not any("solved" in e for e in events)


def test_weak_screen_then_quagmire_no_custom_keywords():
    events, calls = [], []
    def quagmire(*args, **kwargs):
        calls.append(kwargs)
        return {"solver": "quagmire3_shotgun_rust", "top_candidates": [candidate(
            metadata={"alphabet_keyword": "ABCDEF", "cycleword": "AAAA"})]}
    probe.probe_search({"ciphertext": "A" * 200, "language": "en", "periods": [4]}, events.append,
        periodic=lambda *a, **k: {"top_candidates": []}, quagmire=quagmire,
        validator=lambda *a, **k: validation())
    assert calls == [{"language": "en", "keyword_lengths": [6, 7, 8], "cycleword_lengths": [4],
        "hillclimbs": 5000, "restarts": 250, "threads": 4, "seed": 61001,
        "slip_probability": .001, "backtrack_probability": .15, "initial_keywords": [], "top_n": 6}]
    assert events[-1]["candidates"][0]["delivery_eligible"]
    assert probe.selected_step(events[-1]["candidates"][0])["cycleword"] == "AAAA"


def test_invalid_replay_never_validated_and_firewall():
    def never(*a, **k):
        raise AssertionError("invalid replay cannot reach language validator")
    events = []
    probe.probe_search({"ciphertext": "A" * 200, "language": "en", "periods": [4]}, events.append,
        periodic=lambda *a, **k: {"top_candidates": [candidate(plaintext="B" * 200)]},
        quagmire=lambda *a, **k: {"top_candidates": []}, validator=never)
    assert not events[-1]["candidates"][0]["delivery_eligible"]
    with pytest.raises(ValueError, match="allowlist"):
        probe.probe_search({"ciphertext": "A" * 200, "language": "en", "periods": [4],
                            "ground_truth": "secret"}, events.append)


def test_bounded_distinct_retention():
    rows = [dict(candidate(plaintext=chr(65 + i) * 200), content_hash=str(i),
                 validation=validation(validation_score=i), delivery_eligible=False) for i in range(12)]
    rows[0]["delivery_eligible"] = True
    menu = probe.retain(rows + rows)
    assert len(menu) == 6 and len({r["content_hash"] for r in menu}) == 6
    assert menu[0] is rows[0]
    assert [r["content_hash"] for r in probe.retain(rows[::-1])] == [r["content_hash"] for r in menu]


@pytest.mark.parametrize("status,clean,selected", [("completed", True, True), ("timeout", True, False),
    ("worker_error", True, False), ("cleanup_failed", False, False)])
def test_partial_events_preserved_but_only_clean_completion_delivers(monkeypatch, status, clean, selected):
    row = candidate(engine="periodic", validation=validation(), solver="periodic_polyalphabetic_screen")
    def fake(command, request, **kwargs):
        assert kwargs["wall_seconds"] == 45
        assert set(request) == {"ciphertext", "language", "periods"}
        return {"execution_status": status, "cleanup_confirmed": clean,
                "events": [{"event": "finished", "candidates": [row]}]}
    monkeypatch.setattr(probe, "run_guarded", fake)
    result = probe.run_probe(cipher(), {"periods": [4]})
    assert bool(result["selected"]) is selected
    assert len(result["candidates"]) == 1


def test_clean_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-copy")
    monkeypatch.setenv("PYTHONPATH", "untrusted")
    monkeypatch.setenv("DECIPHER_QUAGMIRE_CALIBRATION_KEYWORD", "SECRET")
    monkeypatch.setenv("DECIPHER_NGRAM_MODEL_EN", "/local/model.bin")
    env = probe.probe_environment()
    assert "OPENAI_API_KEY" not in env and "DECIPHER_QUAGMIRE_CALIBRATION_KEYWORD" not in env
    assert env["PYTHONPATH"] != "untrusted"
    assert env["DECIPHER_NGRAM_MODEL_EN"] == "/local/model.bin"


@pytest.mark.parametrize("mode,status", [("off", "skip"), ("probe_v1", "fallback"),
    ("probe_v1", "selected"), ("probe_v1", "cleanup_failed")])
def test_runner_preserves_fallback_and_does_not_declare(monkeypatch, mode, status):
    monkeypatch.setenv("DECIPHER_PERIODIC_ROUTING", mode)
    monkeypatch.setattr(runner, "_select_solver_path", lambda *a, **k: {
        "route": "periodic_polyalphabetic", "solver": "fallback", "reason": "test"})
    monkeypatch.setattr(probe, "diagnose", lambda *a, **k: {"periods": [4], "reason": "test"})
    calls = []
    def fallback(*a, **k):
        calls.append("fallback")
        return "fallback", {}, "FALLBACK", {"name": "fake"}
    monkeypatch.setattr(runner, "_run_periodic_polyalphabetic", fallback)
    row = candidate(engine="periodic", solver="periodic_polyalphabetic_screen")
    def fake_probe(*a, **k):
        assert mode != "off"
        return {"name": "probe_periodic_routing", "execution": {"cleanup_confirmed": status != "cleanup_failed"},
                "decision": "fallback", "selected": row if status == "selected" else None}
    monkeypatch.setattr(probe, "run_probe", fake_probe)
    result = runner.run_automated(cipher(), ground_truth="LABEL_MUST_NOT_REACH_PROBE")
    if status == "cleanup_failed":
        assert result.status == "error" and not calls and not result.final_decryption
    elif status == "selected":
        assert result.status == "completed" and result.final_decryption == row["plaintext"] and not calls
        assert result.artifact["key"] == {}
    else:
        assert result.final_decryption == "FALLBACK" and calls == ["fallback"]
    assert result.artifact["decryption"] == result.final_decryption
    assert result.status != "solved"


def test_fake_worker_completion_and_timeout(tmp_path):
    env = probe.probe_environment()
    good = bounded.run_guarded([sys.executable, "-c", 'print(\'{"event":"done"}\')'], {},
        env=env, cwd=tmp_path, wall_seconds=2)
    assert good["execution_status"] == "completed" and good["cleanup_confirmed"]
    slow = bounded.run_guarded([sys.executable, "-c", 'import time; print(\'{"event":"partial"}\',flush=True); time.sleep(10)'], {},
        env=env, cwd=tmp_path, wall_seconds=.5)
    assert slow["execution_status"] == "timeout" and slow["cleanup_confirmed"]
    assert slow["events"] == [{"event": "partial"}]
    assert slow["wall_seconds"] < 4


def test_cleanup_permission_failure_is_not_success(monkeypatch):
    def denied(*a):
        raise PermissionError("denied")
    monkeypatch.setattr(os, "killpg", denied)
    clean, error = bounded.cleanup_group(SimpleNamespace(pid=12345))
    assert not clean and "PermissionError" in error


def test_outer_timeout_waits_for_nested_guard_cleanup(tmp_path):
    program = (
        "import os,sys; from automated.bounded_process import run_guarded; "
        "run_guarded([sys.executable,'-c','import time; time.sleep(10)'], {}, "
        "env=dict(os.environ), cwd=os.getcwd(), wall_seconds=3)"
    )
    result = bounded.run_guarded([sys.executable, "-c", program], {},
        env=probe.probe_environment(), cwd=tmp_path, wall_seconds=.8)
    assert result["execution_status"] == "timeout"
    assert result["cleanup_confirmed"]
    assert result["wall_seconds"] < 6


def test_missing_native_preserves_published_ordinary_menu():
    events = []
    def unavailable(*a, **k):
        raise RuntimeError("native unavailable")
    with pytest.raises(RuntimeError, match="native unavailable"):
        probe.probe_search({"ciphertext": "A" * 200, "language": "en", "periods": [4]}, events.append,
            periodic=lambda *a, **k: {"top_candidates": [candidate()], "solver": "periodic_polyalphabetic_screen"},
            quagmire=unavailable, validator=lambda *a, **k: validation(validation_label="weak_word_islands"))
    assert any(e["event"] == "menu" for e in events)
    assert events[-1] == {"event": "stage", "engine": "quagmire3"}


def test_quagmire_conflicting_mode_state_rejected():
    row = candidate(engine="quagmire3", metadata={"alphabet_keyword": "ABCDEF", "cycleword": "AAAA",
                                                 "plaintext_alphabet": "ZYXWVUTSRQPONMLKJIHGFEDCBA"})
    assert not probe.replay_matches("A" * 200, row)


def test_closed_caller_lease_interrupts_worker(tmp_path):
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    try:
        result = bounded.supervise([sys.executable, "-c", "import time; time.sleep(10)"], {},
            lease_fd=read_fd, deadline=bounded.time.monotonic() + 2, cwd=tmp_path)
    finally:
        os.close(read_fd)
    assert result["execution_status"] == "interrupted" and result["cleanup_confirmed"]
