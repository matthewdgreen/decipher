"""Tests for the language-model variant registry and its plumbing.

Covers Part 4 of the model-variant-registry spec:
  * Registry: scan / labels / missing-sidecar tolerance; resolution precedence
    (env > variant > default) with named English and German defaults.
  * Runner: ``model_variant`` threading, the ``"auto"`` source mapping, and the
    default-None resolution staying byte-identical.
  * Tools: observe -> set -> search resolves the selected variant path; an
    invalid variant returns a structured error; no paths/shas leak into the
    model-visible tool JSON.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import automated.runner as automated_runner
from analysis import model_registry as mr
from analysis.model_registry import ModelVariantError
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace

REAL_MODELS_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "models"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_model(models_dir, name, meta):
    """Create a fake ``ngram5_<name>.bin`` + sidecar. ``meta=None`` = no sidecar."""
    bin_path = models_dir / f"{name}.bin"
    bin_path.write_bytes(b"fake-model")
    if meta is not None:
        sidecar = models_dir / f"{name}.bin.metadata.json"
        sidecar.write_text(json.dumps(meta), encoding="utf-8")
    return bin_path


def _sidecar(language, variant, label, distinct=100, chars=1000, sha="deadbeef"):
    return {
        "language": language,
        "variant": variant,
        "display_label": label,
        "sha256": sha,
        "corpus_stats": {"distinct_seen_5grams": distinct, "normalized_characters": chars},
    }


# ---------------------------------------------------------------------------
# Registry scan / labels / tolerance
# ---------------------------------------------------------------------------

def test_scan_reads_labels_and_stats(tmp_path):
    _write_model(tmp_path, "ngram5_de", _sidecar("de", "literary_19c", "German lit", 42, 999))
    _write_model(tmp_path, "ngram5_de_dta", _sidecar("de", "historical_1600_1899", "German DTA", 7, 5))
    infos = mr.list_language_models("de", models_dir=tmp_path)
    by_variant = {m.variant: m for m in infos}
    assert set(by_variant) == {"literary_19c", "historical_1600_1899"}
    assert by_variant["literary_19c"].display_label == "German lit"
    assert by_variant["literary_19c"].distinct_ngrams == 42
    assert by_variant["literary_19c"].chars == 999
    assert by_variant["literary_19c"].sha256 == "deadbeef"


def test_language_filter_excludes_other_languages(tmp_path):
    _write_model(tmp_path, "ngram5_de", _sidecar("de", "literary_19c", "German"))
    _write_model(tmp_path, "ngram5_en", _sidecar("en", "gutenberg", "English"))
    de = mr.list_language_models("de", models_dir=tmp_path)
    assert [m.variant for m in de] == ["literary_19c"]
    allm = mr.list_language_models(None, models_dir=tmp_path)
    assert len(allm) == 2


def test_missing_sidecar_is_tolerated(tmp_path):
    _write_model(tmp_path, "ngram5_de", None)  # no sidecar
    infos = mr.list_language_models("de", models_dir=tmp_path)
    assert len(infos) == 1
    info = infos[0]
    assert info.variant is None
    assert info.display_label == "ngram5_de.bin"
    assert info.distinct_ngrams is None and info.sha256 is None


def test_malformed_sidecar_is_tolerated(tmp_path):
    _write_model(tmp_path, "ngram5_de", None)
    (tmp_path / "ngram5_de.bin.metadata.json").write_text("{not valid json", encoding="utf-8")
    infos = mr.list_language_models("de", models_dir=tmp_path)
    assert len(infos) == 1
    assert infos[0].variant is None
    assert infos[0].display_label == "ngram5_de.bin"


def test_scan_missing_dir_returns_empty(tmp_path):
    assert mr.list_language_models("de", models_dir=tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# Resolution precedence: env > variant > default
# ---------------------------------------------------------------------------

def test_resolve_default_uses_language_default_variant(tmp_path):
    """FLIP: `de` has a default variant (historical_1600_1899 -> DTA); the
    default resolution now returns that model, not the bare ngram5_de.bin."""
    _write_model(tmp_path, "ngram5_de", _sidecar("de", "literary_19c", "German"))
    dta = _write_model(tmp_path, "ngram5_de_dta", _sidecar("de", "historical_1600_1899", "DTA"))
    assert mr.resolve_language_model("de", models_dir=tmp_path) == dta


def test_resolve_default_returns_base_model_for_untabled_language(tmp_path):
    """A language absent from _DEFAULT_VARIANTS keeps the bare-filename default,
    even when a historical_1600_1899 variant happens to exist for it."""
    base = _write_model(tmp_path, "ngram5_fr", _sidecar("fr", "literary_19c", "French"))
    _write_model(tmp_path, "ngram5_fr_dta", _sidecar("fr", "historical_1600_1899", "French DTA"))
    assert mr.resolve_language_model("fr", models_dir=tmp_path) == base


def test_default_variant_missing_falls_through_to_base(tmp_path):
    """If the `de` default variant's model is absent, default resolution falls
    through to ngram5_de.bin without raising (a default never raises); and
    active_selection reports plain `default`, not `default_variant`."""
    base = _write_model(tmp_path, "ngram5_de", _sidecar("de", "literary_19c", "German"))
    # No ngram5_de_dta.bin present in this models dir.
    assert mr.resolve_language_model("de", models_dir=tmp_path) == base
    active = mr.active_selection("de", models_dir=tmp_path)
    assert active["source"] == "default"
    assert active["variant"] == "literary_19c"


def test_resolve_variant_selects_matching_slug(tmp_path):
    _write_model(tmp_path, "ngram5_de", _sidecar("de", "literary_19c", "German"))
    dta = _write_model(tmp_path, "ngram5_de_dta", _sidecar("de", "historical_1600_1899", "DTA"))
    assert mr.resolve_language_model("de", "historical_1600_1899", models_dir=tmp_path) == dta


def test_resolve_unknown_variant_raises_listing_available(tmp_path):
    _write_model(tmp_path, "ngram5_de", _sidecar("de", "literary_19c", "German"))
    _write_model(tmp_path, "ngram5_de_dta", _sidecar("de", "historical_1600_1899", "DTA"))
    with pytest.raises(ModelVariantError) as exc:
        mr.resolve_language_model("de", "does_not_exist", models_dir=tmp_path)
    assert exc.value.available == ["historical_1600_1899", "literary_19c"]
    assert "does_not_exist" in str(exc.value)


def test_env_override_wins_over_variant_and_default(tmp_path, monkeypatch):
    _write_model(tmp_path, "ngram5_de", _sidecar("de", "literary_19c", "German"))
    _write_model(tmp_path, "ngram5_de_dta", _sidecar("de", "historical_1600_1899", "DTA"))
    env_model = tmp_path / "override.bin"
    env_model.write_bytes(b"x")
    monkeypatch.setenv("DECIPHER_NGRAM_MODEL_DE", str(env_model))
    # env wins even when an explicit variant is requested (back-compat pin).
    assert mr.resolve_language_model("de", "historical_1600_1899", models_dir=tmp_path) == env_model
    assert mr.resolve_language_model("de", models_dir=tmp_path) == env_model


def test_precedence_demonstration_env_variant_default(tmp_path, monkeypatch):
    """The spec's headline: env > variant > default on `de`, after the DTA flip.

    The load-bearing property: env override and explicit-variant selection both
    still win, even though the *default* now points at the DTA model.
    """
    base = _write_model(tmp_path, "ngram5_de", _sidecar("de", "literary_19c", "German"))
    dta = _write_model(tmp_path, "ngram5_de_dta", _sidecar("de", "historical_1600_1899", "DTA"))
    # default now resolves to the DTA model (the `de` default variant).
    assert mr.resolve_language_model("de", models_dir=tmp_path) == dta
    # An explicit variant still wins over the default: literary_19c selects the
    # OLD Gutenberg model back; historical_1600_1899 matches the new default path.
    assert mr.resolve_language_model("de", "literary_19c", models_dir=tmp_path) == base
    assert mr.resolve_language_model("de", "historical_1600_1899", models_dir=tmp_path) == dta
    # env beats variant AND default.
    env_model = tmp_path / "env.bin"
    env_model.write_bytes(b"x")
    monkeypatch.setenv("DECIPHER_NGRAM_MODEL_DE", str(env_model))
    assert mr.resolve_language_model("de", "literary_19c", models_dir=tmp_path) == env_model
    assert mr.resolve_language_model("de", models_dir=tmp_path) == env_model


def test_non_english_missing_default_returns_none(tmp_path):
    assert mr.resolve_language_model("de", models_dir=tmp_path) is None


# ---------------------------------------------------------------------------
# Back-compat pin: default resolution byte-identical against the real models.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["la", "fr", "it"])
def test_default_resolution_pins_base_model(lang, monkeypatch):
    """Every language WITHOUT a _DEFAULT_VARIANTS entry still resolves the bare
    ngram5_<lang>.bin (byte-identical to before the DTA flip). `de` is covered
    separately by test_default_resolution_pins_de_to_dta."""
    monkeypatch.delenv(f"DECIPHER_NGRAM_MODEL_{lang.upper()}", raising=False)
    resolved = mr.resolve_language_model(lang, models_dir=REAL_MODELS_DIR)
    assert resolved is not None
    assert resolved.name == f"ngram5_{lang}.bin"


def test_default_resolution_pins_en_to_upstream_zenith(monkeypatch):
    """English defaults to the packaged unchanged Zenith 2026.2 model."""
    monkeypatch.delenv("DECIPHER_NGRAM_MODEL_EN", raising=False)
    resolved = mr.resolve_language_model("en", models_dir=REAL_MODELS_DIR)
    assert resolved is not None
    assert resolved.name == "ngram5_en_zenith.bin"


def test_default_resolution_pins_de_to_dta(monkeypatch):
    """FLIP (replaces the prior `de default = ngram5_de.bin` pin): with no env
    and no explicit variant, `de` now resolves to the DTA historical model."""
    monkeypatch.delenv("DECIPHER_NGRAM_MODEL_DE", raising=False)
    resolved = mr.resolve_language_model("de", models_dir=REAL_MODELS_DIR)
    assert resolved is not None
    assert resolved.name == "ngram5_de_dta.bin"


def test_runner_delegates_default_byte_identical(monkeypatch):
    """runner._zenith_native_model_path (no variant) == registry default path."""
    for lang in ["en", "de", "la", "fr", "it", "xx"]:
        monkeypatch.delenv(f"DECIPHER_NGRAM_MODEL_{lang.upper()}", raising=False)
    automated_runner.set_active_model_variant(None)
    for lang in ["en", "de", "la", "fr", "it", "xx"]:
        a = automated_runner._zenith_native_model_path(lang)
        b = mr.resolve_language_model(lang, models_dir=REAL_MODELS_DIR)
        assert (a and os.path.realpath(a)) == (b and os.path.realpath(b)), lang


# ---------------------------------------------------------------------------
# Runner: model_variant threading + "auto" source mapping.
# ---------------------------------------------------------------------------

def test_run_automated_sets_and_restores_active_variant(monkeypatch):
    seen = {}

    def fake_impl(*_args, **kwargs):
        seen["active"] = automated_runner._get_active_model_variant()
        seen["kw"] = kwargs.get("model_variant")
        return automated_runner.AutomatedRunResult(
            test_id="t", status="completed", final_decryption="", elapsed_seconds=0.0
        )

    monkeypatch.setattr(automated_runner, "_run_automated_impl", fake_impl)
    automated_runner.set_active_model_variant(None)
    alpha = Alphabet.from_text("ABC", ignore_chars=set())
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    automated_runner.run_automated(ct, language="de", model_variant="historical_1600_1899")
    assert seen["active"] == "historical_1600_1899"
    assert seen["kw"] == "historical_1600_1899"
    # Restored after the run.
    assert automated_runner._get_active_model_variant() is None


def test_active_variant_slot_is_thread_local(monkeypatch):
    """Two concurrent run_automated calls on different threads must each see
    their own variant at the resolution site (the slot is a threading.local)."""
    import threading

    barrier = threading.Barrier(2, timeout=10)
    recorded: dict[str, str | None] = {}
    errors: list[BaseException] = []

    def spy_resolve(language, variant=None, models_dir=None):
        recorded[threading.current_thread().name] = variant
        return None

    monkeypatch.setattr(
        automated_runner.model_registry, "resolve_language_model", spy_resolve
    )

    def fake_impl(*_args, **_kwargs):
        # Both threads are inside their run (slot set) at the same time.
        barrier.wait()
        automated_runner._zenith_native_model_path("de")  # reads this thread's slot
        barrier.wait()
        return automated_runner.AutomatedRunResult(
            test_id="t", status="completed", final_decryption="", elapsed_seconds=0.0
        )

    monkeypatch.setattr(automated_runner, "_run_automated_impl", fake_impl)

    alpha = Alphabet.from_text("ABC", ignore_chars=set())
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)

    def worker(variant):
        try:
            automated_runner.run_automated(ct, language="de", model_variant=variant)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=("historical_1600_1899",), name="vt1")
    t2 = threading.Thread(target=worker, args=("literary_19c",), name="vt2")
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, errors
    assert recorded == {"vt1": "historical_1600_1899", "vt2": "literary_19c"}
    # The main thread's slot was never touched by the workers.
    assert automated_runner._get_active_model_variant() is None


def test_resolve_model_variant_auto_maps_copiale():
    assert automated_runner.resolve_model_variant("auto", "copiale") == "historical_1600_1899"
    assert automated_runner.resolve_model_variant("auto", "borg") is None
    assert automated_runner.resolve_model_variant(None, "copiale") is None
    assert automated_runner.resolve_model_variant("literary_19c", "borg") == "literary_19c"


def test_resolve_model_variant_auto_is_language_gated():
    """`auto` applies the source mapping only when the run language matches the
    mapping's language — copiale forced to Latin must NOT get the German DTA."""
    assert automated_runner.resolve_model_variant("auto", "copiale", "de") == "historical_1600_1899"
    assert automated_runner.resolve_model_variant("auto", "copiale", "la") is None
    # No language supplied → mapping applies (source implies the language).
    assert automated_runner.resolve_model_variant("auto", "copiale", None) == "historical_1600_1899"
    # An explicit slug is never language-gated (the user's explicit choice).
    assert automated_runner.resolve_model_variant("literary_19c", "copiale", "la") == "literary_19c"


def test_benchmark_runner_auto_threads_resolved_variant(tmp_path, monkeypatch):
    captured = {}

    def fake_run_automated(**kwargs):
        captured["model_variant"] = kwargs.get("model_variant")
        return automated_runner.AutomatedRunResult(
            test_id=kwargs["cipher_id"], status="completed", final_decryption="",
            elapsed_seconds=0.0, run_id="run123", artifact={"key": {}, "steps": []},
        )

    monkeypatch.setattr(automated_runner, "run_automated", fake_run_automated)

    runner = automated_runner.AutomatedBenchmarkRunner(
        artifact_dir=tmp_path, language="de", model_variant="auto"
    )
    test_data = SimpleNamespace(
        canonical_transcription="S001 S002 S003 | S004 S005",
        plaintext="ABCDE",
        plaintext_language="de",
        solver_hints=None,
        transform_pipeline=None,
        test=SimpleNamespace(test_id="copiale_test_1", cipher_system="homophonic", description="d"),
    )
    runner.run_test(test_data)
    assert captured["model_variant"] == "historical_1600_1899"


def test_benchmark_runner_default_none_threads_none(tmp_path, monkeypatch):
    captured = {}

    def fake_run_automated(**kwargs):
        captured["model_variant"] = kwargs.get("model_variant")
        return automated_runner.AutomatedRunResult(
            test_id=kwargs["cipher_id"], status="completed", final_decryption="",
            elapsed_seconds=0.0, run_id="r", artifact={"key": {}, "steps": []},
        )

    monkeypatch.setattr(automated_runner, "run_automated", fake_run_automated)
    runner = automated_runner.AutomatedBenchmarkRunner(artifact_dir=tmp_path, language="de")
    test_data = SimpleNamespace(
        canonical_transcription="S001 S002 | S003",
        plaintext="ABC", plaintext_language="de", solver_hints=None,
        transform_pipeline=None,
        test=SimpleNamespace(test_id="copiale_x", cipher_system="homophonic", description="d"),
    )
    runner.run_test(test_data)
    assert captured["model_variant"] is None


def test_artifact_records_variant_via_sidecar_metadata():
    """binary_ngram_model records the sidecar variant/label for a real model."""
    resolved = automated_runner._zenith_native_model_path("de", variant="historical_1600_1899")
    assert resolved is not None and resolved.name == "ngram5_de_dta.bin"
    automated_runner._zenith_native_model_metadata.cache_clear()
    meta = automated_runner._zenith_native_model_metadata(str(resolved))
    assert meta["variant"] == "historical_1600_1899"
    assert meta["display_label"].startswith("German")


# ---------------------------------------------------------------------------
# Agent tools: observe -> set -> search; invalid variant; no-leak.
# ---------------------------------------------------------------------------

def _executor(language="de"):
    from agent.tools_v2 import WorkspaceToolExecutor

    raw = "S001 S002 S003 S001 S002"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    return WorkspaceToolExecutor(
        workspace=Workspace(ct), language=language,
        word_set={"DER", "UND"}, word_list=["DER", "UND"], pattern_dict={},
    )


def test_observe_language_models_lists_and_reports_active():
    ex = _executor("de")
    out = ex._tool_observe_language_models({})
    variants = {m["variant"] for m in out["models"]}
    assert "historical_1600_1899" in variants and "literary_19c" in variants
    # After the DTA flip the `de` default resolves to the historical variant;
    # the source is reported as `default_variant` so an agent understands WHY a
    # variant is active without having selected one.
    assert out["active"]["source"] == "default_variant"
    assert out["active"]["variant"] == "historical_1600_1899"


def test_act_set_model_variant_updates_selection_and_search_uses_it(monkeypatch):
    ex = _executor("de")
    res = ex._tool_act_set_model_variant({"variant": "historical_1600_1899"})
    assert res["status"] == "ok"
    assert ex._model_variant == "historical_1600_1899"
    assert res["active"]["variant"] == "historical_1600_1899"

    # observe now reports the executor selection as active.
    out = ex._tool_observe_language_models({})
    assert out["active"]["source"] == "variant"
    assert out["active"]["variant"] == "historical_1600_1899"

    # A search tool resolves the model through the selected variant.
    import agent.tools_v2 as tools_v2

    calls = {}

    def spy_path(language, variant=None):
        calls["language"] = language
        calls["variant"] = variant
        return None  # short-circuit the solver; we only assert threading

    monkeypatch.setattr(tools_v2.automated_runner, "_zenith_native_model_path", spy_path)
    ex._tool_search_homophonic_anneal({"branch": "main"})
    assert calls["variant"] == "historical_1600_1899"


def test_act_set_model_variant_invalid_returns_structured_error():
    ex = _executor("de")
    res = ex._tool_act_set_model_variant({"variant": "totally_bogus"})
    assert res["reason"] == "unknown_variant"
    assert "historical_1600_1899" in res["available_variants"]
    assert ex._model_variant is None  # unchanged on failure


def test_registry_tool_outputs_do_not_leak_paths_or_shas():
    ex = _executor("de")
    observe = json.dumps(ex._tool_observe_language_models({}), default=str)
    ex._tool_act_set_model_variant({"variant": "historical_1600_1899"})
    setres = json.dumps(ex._tool_act_set_model_variant({"variant": "literary_19c"}), default=str)
    for blob in (observe, setres):
        assert ".bin" not in blob, "a model path leaked into tool JSON"
        assert "/models/" not in blob and "\\models\\" not in blob
        # No 64-hex sha strings.
        assert not any(
            len(tok) >= 40 and all(c in "0123456789abcdef" for c in tok.lower())
            for tok in blob.replace('"', " ").replace(",", " ").split()
        ), "a sha256 leaked into tool JSON"


def test_act_set_model_variant_notes_with_and_without_env_override(monkeypatch, tmp_path):
    """Without an env override the note describes an effective switch; with an
    override the response stays ok but warns the selection is inert."""
    # Normal case: no env override → effective note, no warning.
    monkeypatch.delenv("DECIPHER_NGRAM_MODEL_DE", raising=False)
    ex = _executor("de")
    res = ex._tool_act_set_model_variant({"variant": "historical_1600_1899"})
    assert res["status"] == "ok"
    assert "warning" not in res
    assert "will now use this variant" in res["note"]

    # Env-override case: selection recorded but INERT (env wins everywhere).
    env_model = tmp_path / "override_de.bin"
    env_model.write_bytes(b"x")
    monkeypatch.setenv("DECIPHER_NGRAM_MODEL_DE", str(env_model))
    ex2 = _executor("de")
    res2 = ex2._tool_act_set_model_variant({"variant": "historical_1600_1899"})
    assert res2["status"] == "ok"
    assert res2["warning"] == "environment_override_active"
    assert "INERT" in res2["note"]
    assert "DECIPHER_NGRAM_MODEL_DE" in res2["note"]
    # The selection IS recorded (it becomes effective once the override is unset).
    assert ex2._model_variant == "historical_1600_1899"


# ---------------------------------------------------------------------------
# v3 propagation: episode inheritance + state round-trip.
# ---------------------------------------------------------------------------

def test_episode_executor_inherits_lead_model_variant(monkeypatch):
    """run_episode seeds each fresh episode executor from state.model_variant."""
    import investigation.episodes as episodes_mod
    from agent.model_provider import ModelResponse, ModelUsage, ToolUseBlock
    from investigation.episodes import EpisodeSpec, run_episode
    from investigation.sessions import SessionCapabilities
    from investigation.state import BudgetEntry, InvestigationState

    created: list[str | None] = []
    real_executor_cls = episodes_mod.WorkspaceToolExecutor

    class SpyExecutor(real_executor_cls):
        def __init__(self, *args, **kwargs):
            created.append(kwargs.get("model_variant"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(episodes_mod, "WorkspaceToolExecutor", SpyExecutor)

    class _Session:
        model = "fake"
        provider_name = "openai"
        capabilities = SessionCapabilities()

        def __init__(self):
            self._budget = []

        def send(self, blocks, tools=None, max_tokens=8192):
            self._budget.append(BudgetEntry("episode:survey", "openai", "fake", 10, 2, 0))
            good = {"findings": ["f"], "suspected_modes": [], "recommended_next": []}
            return ModelResponse(
                content=[ToolUseBlock(id="s1", name="episode_submit_result",
                                      input={"result": good, "summary": "done"})],
                usage=ModelUsage(10, 2, 0),
            )

        def usage_entries(self):
            return list(self._budget)

        def export_transcript(self):
            return {"provider": "openai", "model": "fake", "exchanges": []}

    raw = "ABCDEFGHIJKL"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    state = InvestigationState(workspace=Workspace(ct), language="de")
    state.model_variant = "historical_1600_1899"

    spec = EpisodeSpec("survey", "diagnose", inputs={"branches": ["main"]})
    result = run_episode(spec, state, session=_Session())
    assert result.status == "ok"
    assert created == ["historical_1600_1899"]
    # And with no selection, episodes get the default.
    created.clear()
    state.model_variant = None
    run_episode(spec, state, session=_Session())
    assert created == [None]


def test_investigation_state_round_trips_model_variant():
    from investigation.state import InvestigationState

    raw = "ABCDEFG"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    state = InvestigationState(workspace=Workspace(ct), language="de")
    state.model_variant = "historical_1600_1899"

    restored = InvestigationState.from_artifact_dict(state.to_artifact_dict())
    assert restored.model_variant == "historical_1600_1899"

    # Default None round-trips too (and old artifacts without the key restore).
    state.model_variant = None
    data = state.to_artifact_dict()
    assert data["model_variant"] is None
    restored2 = InvestigationState.from_artifact_dict(data)
    assert restored2.model_variant is None
    data.pop("model_variant")  # pre-slice artifact
    restored3 = InvestigationState.from_artifact_dict(data)
    assert restored3.model_variant is None


# ---------------------------------------------------------------------------
# Preflight line: content + v2/v3 context insertion.
# ---------------------------------------------------------------------------

def test_format_registry_preflight_line_content(monkeypatch):
    monkeypatch.delenv("DECIPHER_NGRAM_MODEL_DE", raising=False)
    line = mr.format_registry_preflight_line("de")
    assert line is not None
    for slug in ("literary_19c", "literary_19c_small", "historical_1600_1899"):
        assert slug in line
    # After the DTA flip the default-active variant is the DTA historical model,
    # and the line is honest that this is the language default (not a selection).
    assert "historical_1600_1899 (active)" in line
    assert "[language default]" in line
    assert "act_set_model_variant" in line and "observe_language_models" in line
    # A language with no discoverable models yields None (line omitted).
    assert mr.format_registry_preflight_line("zz") is None
    # An active explicit variant is marked.
    line_dta = mr.format_registry_preflight_line("de", "historical_1600_1899")
    assert "historical_1600_1899 (active)" in line_dta


def test_loop_v2_initial_context_includes_variant_line(monkeypatch):
    """The v2 cipher-diagnostic preflight gains the variant line (additive)."""
    from types import SimpleNamespace as NS

    from agent.loop_v2 import run_v2

    monkeypatch.delenv("DECIPHER_NGRAM_MODEL_DE", raising=False)

    class _API:
        model = "claude-sonnet-4-6"

        def __init__(self):
            self.messages_seen = []
            self.n = 0

        def send_message(self, messages, tools=None, system="", max_tokens=4096):
            self.messages_seen.append(messages)
            self.n += 1
            return NS(
                usage=NS(input_tokens=10, output_tokens=2),
                content=[NS(type="tool_use", id=f"d{self.n}", name="decode_show",
                            input={"branch": "main"})],
            )

    raw = "ABCDEFGHIJABCDEFGHIJABCDEFGHIJ"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    api = _API()
    run_v2(cipher_text=ct, claude_api=api, language="de", max_iterations=2,
           cipher_id="variant_line_test")
    assert api.messages_seen, "provider was never called"
    initial = api.messages_seen[0][0]["content"]
    assert "Language models for de:" in initial
    assert "historical_1600_1899" in initial


def test_v3_fingerprint_section_includes_variant_line(monkeypatch):
    """The v3 context builder's fingerprint section gains the variant line."""
    from investigation import context as v3_context
    from investigation.state import InvestigationState

    monkeypatch.delenv("DECIPHER_NGRAM_MODEL_DE", raising=False)
    raw = "S001 S002 S003 S004 S005 S006 S007 S008"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    state = InvestigationState(workspace=Workspace(ct), language="de")
    section = v3_context._render_fingerprint(state)
    assert "Language models for de:" in section
    assert "historical_1600_1899" in section
