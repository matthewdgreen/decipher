"""Runner integration tests for the Phase-2b word-repair refinement.

Covers the refinement dispatch, the composed gate (spec item 3, FINAL
2026-07-13: menu-only by default — the step records ``would_adopt`` from the
composed gate (library repair-acceptance verdict AND strict
validation_score_v2 improvement) without modifying the run; adoption is
opt-in via DECIPHER_WORD_REPAIR_ADOPT=1), the `DECIPHER_WORD_REPAIR_*` env
surface, and the lazy-import binding constraint. The heavy homophonic solve and
the `propose_word_repairs` menu are stubbed so these tests stay fast and
deterministic.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import automated.runner as runner
from analysis.candidate_packet import CandidatePacket
from models.alphabet import Alphabet
from models.cipher_text import CipherText


def _small_cipher() -> CipherText:
    """Three-symbol page repeated so project_pages yields a real decryption."""
    alpha = Alphabet(["S001", "S002", "S003"])
    tokens = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    raw = " ".join(alpha.symbol_for(t) for t in tokens)
    return CipherText(raw=raw, alphabet=alpha, separator=None)


def _word_repair_packet(
    *,
    page_validation: float,
    edits: list[str],
    accepted: bool = True,
    adjudication_score: float = 1.0,
) -> CandidatePacket:
    """A canned word-repair packet carrying both composed-gate signals.

    ``validation`` is the library's repair_acceptance verdict dict (as
    ``propose_word_repairs`` attaches it); ``solver_scores`` carries the
    rank-key components plus ``adjudication_score``.
    """
    return CandidatePacket(
        candidate_id="edits:" + ",".join(edits),
        kind="word_repair",
        source={"solver": "word_hypothesis_repair"},
        rank=1,
        text=None,
        preview="AAA BBB",
        solver_scores={
            "page_validation_avg": page_validation,
            "adjudication_score": adjudication_score,
        },
        validation={
            "accepted": accepted,
            "decision": "runtime_accept" if accepted else "hold_for_review",
            "reasons": ["canned verdict"],
        },
        provenance={"edits": edits, "mask": []},
    )


@pytest.fixture
def stub_model(monkeypatch):
    """Skip real 47MB model resolution/hashing in unit tests."""
    monkeypatch.setattr(runner, "zenith_native_model_path", lambda _lang: None)


@pytest.fixture
def adopt_enabled(monkeypatch):
    """Opt in to composed-gate adoption (menu-only is the default)."""
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_ADOPT", "1")


@pytest.fixture
def adopt_default(monkeypatch):
    """Guard the default (menu-only) path against ambient env."""
    monkeypatch.delenv("DECIPHER_WORD_REPAIR_ADOPT", raising=False)


def _patch_baseline_validation(monkeypatch, value: float) -> None:
    import analysis.multipage as multipage

    monkeypatch.setattr(
        multipage,
        "score_page_runtime",
        lambda row, *, key, mask, language, model_path=None: {
            "validation_score_v2": value,
            "validation_components_v2": {},
            "language_quality_mean": 0.0,
            "language_quality_features": {},
            "dict_rate": 0.0,
            "diagnostics": {},
            "test_id": row["test_id"],
        },
    )


def _patch_propose(monkeypatch, packets) -> None:
    import analysis.word_hypothesis_repair as whr

    monkeypatch.setattr(whr, "propose_word_repairs", lambda **_kwargs: list(packets))


def test_word_repair_adopts_accepted_verdict_and_improving_validation(monkeypatch, stub_model, adopt_enabled):
    """Composed gate case 1: accepted verdict + improving validation -> adopted."""
    cipher = _small_cipher()
    base_key = {0: 0, 1: 1, 2: 2}  # A, B, C
    base_decryption = "ABCABCABCABC"

    _patch_baseline_validation(monkeypatch, 1.0)
    _patch_propose(
        monkeypatch,
        [_word_repair_packet(page_validation=2.5, edits=["S001:A->X"], accepted=True, adjudication_score=6.7)],
    )

    step, adopted = runner._run_word_repair_refinement(
        cipher_text=cipher,
        language="en",
        refinement="word_repair",
        base_solver="native_homophonic_anneal",
        base_key=base_key,
        base_decryption=base_decryption,
        mask=(),
    )

    assert step["name"] == "search_word_repair"
    assert step["status"] == "completed"
    assert step["mode"] == "word_repair"
    assert step["validation_before"] == 1.0
    assert step["validation_after"] == 2.5
    assert step["validation_delta"] == 1.5
    assert step["counts"] == {
        "proposed": 1,
        "prescreened": 1,
        "adjudicated": 1,
        "improving": 1,
        "verdict_accepted": 1,
        "passed_composed_gate": 1,
        "adopted": 1,
        "rejected": 0,
    }
    assert step["adopted"]["edits"] == ["S001:A->X"]
    assert step["adopted"]["solver"] == "word_repair_homophonic"
    assert step["adopted_reason"] == "composed_gate_passed"
    # Both composed-gate signals are explicit on the adopted entry.
    assert step["adopted"]["acceptance"]["accepted"] is True
    assert step["adopted"]["acceptance"]["decision"] == "runtime_accept"
    assert step["adopted"]["adjudication_score"] == 6.7
    assert step["adopted"]["page_validation_avg"] == 2.5
    # ... and on the per-candidate gate decisions.
    decision = step["gate_decisions"][0]
    assert decision["acceptance_accepted"] is True
    assert decision["validation_improves"] is True
    assert decision["adjudication_score"] == 6.7
    assert decision["passed_composed_gate"] is True
    assert decision["adopted"] is True
    # Candidate menu carries the packet dicts with text=None (F3 deferral).
    assert step["candidate_menu"] and step["candidate_menu"][0]["text"] is None

    assert adopted is not None
    solver, new_key, new_decryption = adopted
    assert solver == "word_repair_homophonic"
    # S001 (token 0) was reassigned A(0) -> X(23); base_key is not mutated.
    assert new_key[0] == ord("X") - ord("A")
    assert base_key[0] == 0
    assert new_decryption == "XBCXBCXBCXBC"


def test_word_repair_rejects_improving_validation_with_rejected_verdict(monkeypatch, stub_model, adopt_enabled):
    """Composed gate case 2 (the first acceptance run's exact regression):
    validation improves but the library's repair-acceptance verdict rejects
    -> NOT adopted."""
    cipher = _small_cipher()
    base_key = {0: 0, 1: 1, 2: 2}

    _patch_baseline_validation(monkeypatch, 1.0)
    _patch_propose(
        monkeypatch,
        [_word_repair_packet(page_validation=2.5, edits=["S001:A->X"], accepted=False, adjudication_score=-3.0)],
    )

    step, adopted = runner._run_word_repair_refinement(
        cipher_text=cipher,
        language="en",
        refinement="word_repair",
        base_solver="native_homophonic_anneal",
        base_key=base_key,
        base_decryption="ABCABCABCABC",
        mask=(),
    )

    assert adopted is None
    assert step["adopted"] is None
    assert step["adopted_reason"] == "no_candidate_passed_composed_gate"
    assert step["validation_delta"] == 0.0
    assert step["counts"]["improving"] == 1
    assert step["counts"]["verdict_accepted"] == 0
    assert step["counts"]["passed_composed_gate"] == 0
    assert step["counts"]["adopted"] == 0
    assert step["counts"]["rejected"] == 1
    # The rejected entry records both signals.
    decision = step["gate_decisions"][0]
    assert decision["validation_improves"] is True
    assert decision["acceptance_accepted"] is False
    assert decision["acceptance_decision"] == "hold_for_review"
    assert decision["adjudication_score"] == -3.0
    assert decision["passed_composed_gate"] is False
    assert decision["adopted"] is False


def test_word_repair_rejects_accepted_verdict_without_validation_improvement(monkeypatch, stub_model, adopt_enabled):
    """Composed gate case 3: accepted verdict but non-improving validation
    -> NOT adopted."""
    cipher = _small_cipher()
    base_key = {0: 0, 1: 1, 2: 2}

    _patch_baseline_validation(monkeypatch, 1.0)
    _patch_propose(
        monkeypatch,
        [_word_repair_packet(page_validation=0.5, edits=["S001:A->X"], accepted=True, adjudication_score=2.0)],
    )

    step, adopted = runner._run_word_repair_refinement(
        cipher_text=cipher,
        language="en",
        refinement="word_repair",
        base_solver="native_homophonic_anneal",
        base_key=base_key,
        base_decryption="ABCABCABCABC",
        mask=(),
    )

    assert adopted is None
    assert step["adopted"] is None
    assert step["adopted_reason"] == "no_candidate_passed_composed_gate"
    assert step["validation_delta"] == 0.0
    assert step["counts"]["improving"] == 0
    assert step["counts"]["verdict_accepted"] == 1
    assert step["counts"]["passed_composed_gate"] == 0
    assert step["counts"]["adopted"] == 0
    assert step["counts"]["rejected"] == 1
    decision = step["gate_decisions"][0]
    assert decision["validation_improves"] is False
    assert decision["acceptance_accepted"] is True
    assert decision["passed_composed_gate"] is False


def test_word_repair_selects_best_gate_passing_candidate(monkeypatch, stub_model, adopt_enabled):
    """Selection runs among gate-passing candidates only: a higher-validation
    candidate with a rejected verdict must not shadow a lower-validation one
    that passes both signals."""
    cipher = _small_cipher()
    base_key = {0: 0, 1: 1, 2: 2}

    _patch_baseline_validation(monkeypatch, 1.0)
    _patch_propose(
        monkeypatch,
        [
            _word_repair_packet(page_validation=3.0, edits=["S002:B->Y"], accepted=False, adjudication_score=-1.0),
            _word_repair_packet(page_validation=2.0, edits=["S001:A->X"], accepted=True, adjudication_score=4.0),
        ],
    )

    step, adopted = runner._run_word_repair_refinement(
        cipher_text=cipher,
        language="en",
        refinement="word_repair",
        base_solver="native_homophonic_anneal",
        base_key=base_key,
        base_decryption="ABCABCABCABC",
        mask=(),
    )

    assert adopted is not None
    assert step["adopted"]["edits"] == ["S001:A->X"]
    assert step["adopted"]["adjudication_score"] == 4.0
    assert step["validation_after"] == 2.0
    assert step["counts"]["improving"] == 2
    assert step["counts"]["verdict_accepted"] == 1
    assert step["counts"]["passed_composed_gate"] == 1
    assert step["counts"]["adopted"] == 1
    assert step["counts"]["rejected"] == 1
    by_edits = {tuple(d["edits"]): d for d in step["gate_decisions"]}
    assert by_edits[("S002:B->Y",)]["adopted"] is False
    assert by_edits[("S001:A->X",)]["adopted"] is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", False),  # empty string fails closed
        ("maybe", False),  # garbage fails closed
        ("0", False),
        ("2", False),  # non-allowlisted truthy-looking value fails closed
        ("1", True),
        ("true", True),
        (" TRUE ", True),  # case/whitespace tolerant
        ("Yes", True),
        ("on", True),
        ("off", False),
    ],
)
def test_word_repair_adopt_flag_strict_parse(monkeypatch, raw, expected):
    """DECIPHER_WORD_REPAIR_ADOPT fails closed: only an explicit affirmative
    value enables adoption; empty/garbage values keep the menu-only default."""
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_ADOPT", raw)
    assert runner._word_repair_adopt_enabled() is expected


def test_word_repair_adopt_flag_unset_is_disabled(monkeypatch):
    monkeypatch.delenv("DECIPHER_WORD_REPAIR_ADOPT", raising=False)
    assert runner._word_repair_adopt_enabled() is False


def test_word_repair_rejects_whole_candidate_on_partial_edit_application(
    monkeypatch, stub_model, adopt_enabled
):
    """If ANY edit label fails to parse/apply, the entire candidate is
    rejected (no_applicable_edits) — never a silent subset adoption."""
    cipher = _small_cipher()
    base_key = {0: 0, 1: 1, 2: 2}

    _patch_baseline_validation(monkeypatch, 1.0)
    _patch_propose(
        monkeypatch,
        [
            _word_repair_packet(
                page_validation=2.5,
                edits=["S001:A->X", "BOGUS LABEL"],  # one good + one bogus
                accepted=True,
                adjudication_score=3.0,
            )
        ],
    )

    step, adopted = runner._run_word_repair_refinement(
        cipher_text=cipher,
        language="en",
        refinement="word_repair",
        base_solver="native_homophonic_anneal",
        base_key=base_key,
        base_decryption="ABCABCABCABC",
        mask=(),
    )

    assert adopted is None
    assert step["adopted"] is None
    assert step["adopted_reason"] == "no_applicable_edits"
    assert step["would_adopt"] is None
    assert step["would_adopt_reason"] == "no_applicable_edits"
    assert step["validation_delta"] == 0.0
    assert base_key == {0: 0, 1: 1, 2: 2}


def test_word_repair_rejects_candidate_editing_masked_symbol(
    monkeypatch, stub_model, adopt_enabled
):
    """A candidate whose edit targets a symbol in the active mask is rejected:
    the library scored it on the masked projection where that symbol never
    appears, so the runner's projected result would diverge from what was
    scored."""
    cipher = _small_cipher()
    base_key = {0: 0, 1: 1, 2: 2}

    _patch_baseline_validation(monkeypatch, 1.0)
    _patch_propose(
        monkeypatch,
        [
            _word_repair_packet(
                page_validation=2.5,
                edits=["S001:A->X"],
                accepted=True,
                adjudication_score=3.0,
            )
        ],
    )

    step, adopted = runner._run_word_repair_refinement(
        cipher_text=cipher,
        language="en",
        refinement="word_repair",
        base_solver="native_homophonic_anneal",
        base_key=base_key,
        base_decryption="BCBCBCBC",
        mask=("S001",),  # the edited symbol is masked
    )

    assert adopted is None
    assert step["adopted"] is None
    assert step["adopted_reason"] == "no_applicable_edits"
    assert step["would_adopt"] is None
    assert step["would_adopt_reason"] == "no_applicable_edits"
    assert base_key == {0: 0, 1: 1, 2: 2}


def test_word_repair_zero_repairs_is_valid_outcome(monkeypatch, stub_model, adopt_default):
    """Zero repairs proposed on the default (menu-only) path is a valid outcome."""
    cipher = _small_cipher()
    _patch_baseline_validation(monkeypatch, 1.0)
    _patch_propose(monkeypatch, [])

    step, adopted = runner._run_word_repair_refinement(
        cipher_text=cipher,
        language="en",
        refinement="word_repair",
        base_solver="native_homophonic_anneal",
        base_key={0: 0, 1: 1, 2: 2},
        base_decryption="ABCABCABCABC",
        mask=(),
    )
    assert adopted is None
    assert step["adopted_reason"] == "menu_only_default"
    assert step["would_adopt"] is None
    assert step["would_adopt_reason"] == "no_repairs_proposed"
    assert step["counts"]["proposed"] == 0


def test_word_repair_menu_only_default_records_would_adopt_without_adopting(
    monkeypatch, stub_model, adopt_default
):
    """Default path: the composed gate is measurement-only. Even a candidate
    that passes both signals is recorded as would_adopt, never applied."""
    cipher = _small_cipher()
    base_key = {0: 0, 1: 1, 2: 2}

    _patch_baseline_validation(monkeypatch, 1.0)
    _patch_propose(
        monkeypatch,
        [_word_repair_packet(page_validation=2.5, edits=["S001:A->X"], accepted=True, adjudication_score=6.7)],
    )

    step, adopted = runner._run_word_repair_refinement(
        cipher_text=cipher,
        language="en",
        refinement="word_repair",
        base_solver="native_homophonic_anneal",
        base_key=base_key,
        base_decryption="ABCABCABCABC",
        mask=(),
    )

    # Nothing is adopted and the run is not modified.
    assert adopted is None
    assert step["adopted"] is None
    assert step["adopted_reason"] == "menu_only_default"
    assert step["adopt_enabled"] is False
    assert step["validation_delta"] == 0.0
    assert base_key == {0: 0, 1: 1, 2: 2}
    assert step["counts"]["passed_composed_gate"] == 1
    assert step["counts"]["adopted"] == 0
    assert step["counts"]["rejected"] == 1
    # The gate's selection is still measured: would_adopt shape pinned.
    would = step["would_adopt"]
    assert set(would.keys()) == {
        "edits",
        "candidate_id",
        "preview",
        "acceptance",
        "adjudication_score",
        "page_validation_avg",
    }
    assert would["edits"] == ["S001:A->X"]
    assert would["candidate_id"] == "edits:S001:A->X"
    assert would["acceptance"] == {
        "accepted": True,
        "decision": "runtime_accept",
        "reasons": ["canned verdict"],
    }
    assert would["adjudication_score"] == 6.7
    assert would["page_validation_avg"] == 2.5
    assert step["would_adopt_reason"] == "composed_gate_passed"
    # Menu and gate decisions are still recorded in full.
    assert len(step["candidate_menu"]) == 1
    decision = step["gate_decisions"][0]
    assert decision["passed_composed_gate"] is True
    assert decision["adopted"] is False


def test_word_repair_default_path_run_automated_is_no_op(monkeypatch, stub_model, adopt_default):
    """End-to-end default path: run_automated with word_repair leaves the final
    key/decryption exactly equal to the pre-refinement result while still
    recording the menu step."""
    cipher = _small_cipher()

    def fake_homophonic(cipher_text, language, **_kwargs):
        step = {"name": "search_homophonic_anneal", "solver": "native_homophonic_anneal"}
        return "native_homophonic_anneal", {0: 0, 1: 1, 2: 2}, "ABCABCABCABC", step

    monkeypatch.setattr(runner, "_run_homophonic", fake_homophonic)
    _patch_baseline_validation(monkeypatch, 1.0)
    _patch_propose(
        monkeypatch,
        [_word_repair_packet(page_validation=2.5, edits=["S001:A->X"], accepted=True, adjudication_score=6.7)],
    )

    result = runner.run_automated(
        cipher_text=cipher,
        language="en",
        cipher_id="menu_only_test",
        cipher_system="homophonic_substitution",
        homophonic_refinement="word_repair",
    )

    # Final result equals the pre-refinement solve.
    assert result.final_decryption == "ABCABCABCABC"
    assert result.solver == "native_homophonic_anneal"
    assert result.artifact["key"] == {"0": 0, "1": 1, "2": 2}
    # The menu step is still recorded, with the gate's would-adopt selection.
    step = next(s for s in result.steps if s.get("name") == "search_word_repair")
    assert step["adopted"] is None
    assert step["adopted_reason"] == "menu_only_default"
    assert step["would_adopt"]["edits"] == ["S001:A->X"]


@pytest.mark.parametrize("refinement", ["word_repair", "null_masks+word_repair"])
def test_dispatch_reaches_word_repair_path(monkeypatch, refinement):
    """run_automated routes both new values through _run_word_repair_refinement."""
    cipher = _small_cipher()

    def fake_homophonic(cipher_text, language, **_kwargs):
        step = {"name": "search_homophonic_anneal", "solver": "native_homophonic_anneal"}
        return "native_homophonic_anneal", {0: 0, 1: 1, 2: 2}, "ABCABCABCABC", step

    def fake_bakeoff(**_kwargs):
        return {
            "name": "search_null_masks",
            "status": "completed",
            "selected": {"status": "completed", "key": {"0": 0, "1": 1, "2": 2}, "mask": [], "decryption": "ABCABCABCABC"},
        }

    calls: list[dict] = []

    def fake_word_repair(**kwargs):
        calls.append(kwargs)
        step = {"name": "search_word_repair", "status": "completed", "mode": kwargs["refinement"]}
        return step, ("word_repair_homophonic", {0: 23, 1: 1, 2: 2}, "XBCXBCXBCXBC")

    monkeypatch.setattr(runner, "_run_homophonic", fake_homophonic)
    monkeypatch.setattr(runner, "_run_null_mask_bakeoff", fake_bakeoff)
    monkeypatch.setattr(runner, "_run_word_repair_refinement", fake_word_repair)

    result = runner.run_automated(
        cipher_text=cipher,
        language="en",
        cipher_id="dispatch_test",
        cipher_system="homophonic_substitution",
        homophonic_refinement=refinement,
    )

    assert len(calls) == 1, "word-repair refinement was not reached exactly once"
    assert calls[0]["refinement"] == refinement
    # Composite passes the null-mask winner's mask; plain passes ().
    assert calls[0]["mask"] == ()
    step_names = [step.get("name") for step in result.steps]
    assert "search_word_repair" in step_names
    if refinement == "null_masks+word_repair":
        assert "search_null_masks" in step_names
    # The adopted repair is swapped into the final result (artifact key is
    # str-keyed by to_artifact()).
    assert result.artifact["key"] == {"0": 23, "1": 1, "2": 2}
    assert result.final_decryption == "XBCXBCXBCXBC"
    assert result.solver == "word_repair_homophonic"


def test_word_repair_config_env_round_trip(monkeypatch):
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_WINDOW_SIZE", "88")
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_WINDOW_STEP", "22")
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_MIN_WORD_LEN", "4")
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_MAX_WORD_LEN", "12")
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_MAX_EDITS", "2")
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_MAX_HYPOTHESES", "50")
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_MAX_HYPOTHESES_PER_WINDOW", "7")
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_ACCEPTANCE_MARGIN", "0.09")
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_MIN_PAGE_DROP", "0.05")
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_MAX_ILLUSION_INCREASE", "0.04")

    config, env_used = runner._word_repair_config_from_env()
    assert config.window_size == 88
    assert config.window_step == 22
    assert config.min_word_len == 4
    assert config.max_word_len == 12
    assert config.max_edits == 2
    assert config.max_hypotheses == 50
    assert config.max_hypotheses_per_window == 7
    assert config.acceptance_margin == 0.09
    assert config.min_page_drop == 0.05
    assert config.max_illusion_increase == 0.04
    assert env_used["DECIPHER_WORD_REPAIR_WINDOW_SIZE"] == 88
    assert env_used["DECIPHER_WORD_REPAIR_ACCEPTANCE_MARGIN"] == 0.09


def test_word_repair_config_defaults_match_library(monkeypatch):
    for env_name in list(runner._WORD_REPAIR_ENV_INT_FIELDS.values()) + list(
        runner._WORD_REPAIR_ENV_FLOAT_FIELDS.values()
    ):
        monkeypatch.delenv(env_name, raising=False)
    config, env_used = runner._word_repair_config_from_env()
    from analysis.word_hypothesis_repair import WordRepairConfig

    assert config == WordRepairConfig()
    assert env_used == {}


def test_word_repair_config_bad_int_raises(monkeypatch):
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_WINDOW_SIZE", "not-a-number")
    with pytest.raises(ValueError):
        runner._word_repair_config_from_env()


def test_word_repair_config_bad_float_raises(monkeypatch):
    monkeypatch.setenv("DECIPHER_WORD_REPAIR_ACCEPTANCE_MARGIN", "high")
    with pytest.raises(ValueError):
        runner._word_repair_config_from_env()


def test_build_word_repair_menu_wrapper_kwarg_parity(monkeypatch):
    """F3 (Phase-2.4 review): direct coverage for the ``build_word_repair_menu``
    single-page wrapper, whose only production caller is the agent menu tool
    (which every other test monkeypatches). The group-native core is stubbed
    with a kwarg recorder; every kwarg uses a distinct sentinel value, so an
    accidentally swapped pair (e.g. ``mask``<->``language``) fails loudly."""
    cipher = _small_cipher()
    captured: dict = {}
    sentinel_menu = object()

    def fake_core(**kwargs):
        captured.update(kwargs)
        return sentinel_menu

    monkeypatch.setattr(runner, "build_word_repair_menu_for_pages", fake_core)

    base_key = {0: 4, 1: 7, 2: 11}
    config_sentinel = object()
    result = runner.build_word_repair_menu(
        cipher_text=cipher,
        base_key=base_key,
        mask=("S002",),
        language="de",
        config=config_sentinel,
        dictionary_path="/sentinel/dictionary.txt",
        model_path="/sentinel/model.bin",
        source_branch="sentinel_branch",
    )

    # The wrapper returns the core's menu verbatim.
    assert result is sentinel_menu

    # Kwarg parity, each against its own distinct sentinel.
    assert captured["base_key"] == {0: 4, 1: 7, 2: 11}
    assert captured["mask"] == ("S002",)
    assert captured["language"] == "de"
    assert captured["config"] is config_sentinel
    assert captured["dictionary_path"] == "/sentinel/dictionary.txt"
    assert captured["model_path"] == "/sentinel/model.bin"
    assert captured["source_branch"] == "sentinel_branch"
    assert captured["alphabet"] is cipher.alphabet

    # The page group is the shared single-page construction (_single_page_group):
    # one PageBundle mirroring the ciphertext, with plaintext firewalled empty.
    pages = captured["pages"]
    assert len(pages) == 1
    page = pages[0]
    assert page.test_id == "page_0"
    assert page.plaintext == ""
    assert page.token_ids == list(cipher.tokens)
    assert page.symbols == [cipher.alphabet.symbol_for(t) for t in cipher.tokens]


def test_runner_does_not_import_promoted_libraries_at_module_level():
    """Binding constraint 1: lazy imports only.

    A top-level ``import analysis.multipage`` / ``analysis.word_hypothesis_repair``
    in the runner would hit a partially-initialized module (multipage imports
    automated.runner at its top, and the runner's public wrappers sit at EOF).
    Assert both via a source-level AST scan and a fresh-import sys.modules probe.
    """
    source_path = os.path.join(os.path.dirname(__file__), "..", "src", "automated", "runner.py")
    with open(source_path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    banned = {"analysis.multipage", "analysis.word_hypothesis_repair"}
    for node in tree.body:  # module-level statements only
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in banned, f"top-level import of {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in banned, f"top-level from-import of {node.module}"

    # Fresh-import probe in an isolated subprocess: importing the runner must
    # not pull the libraries into sys.modules.
    src_root = os.path.join(os.path.dirname(__file__), "..", "src")
    probe = textwrap.dedent(
        """
        import sys
        import automated.runner  # noqa: F401
        assert "analysis.multipage" not in sys.modules, "multipage imported at runner load"
        assert "analysis.word_hypothesis_repair" not in sys.modules, "whr imported at runner load"
        print("ok")
        """
    )
    env = dict(os.environ, PYTHONPATH=src_root)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
