"""Runner integration tests for the opt-in LLM finalist reranker (Phase 4a 4.2).

Covers the strict env parses, the core ``_finalist_reader_rank`` helper in
annotate vs select modes, graceful no-key degradation, the firewall on the
prompt the runner actually constructs, and end-to-end recording of the
``finalist_reader`` block on the real null-mask bakeoff and word-repair steps.
The model is always stubbed (no network).
"""
from __future__ import annotations

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cli
import automated.runner as runner
import analysis.llm_reader as llm_reader
from agent.model_provider import ModelResponse, ModelUsage, TextBlock
from analysis.candidate_packet import CandidatePacket
from benchmark.loader import parse_canonical_transcription


class _CapturingProvider:
    """Fake provider that records prompts and ranks the LAST candidate first.

    Ranking the last display id first makes the reader's best index non-zero
    whenever there are >= 2 candidates, so selection-override is observable.
    """

    provider_name = "openai"

    def __init__(self, *, model="gpt-5.6-luna", best="last"):
        self.model = model
        self.best = best
        self.sent_prompts: list[str] = []

    def send(self, *, messages, tools=None, system="", max_tokens=4096):
        prompt = messages[-1]["content"]
        self.sent_prompts.append(prompt)
        ids = re.findall(r"^(C\d\d):$", prompt, flags=re.MULTILINE)
        if self.best == "last":
            ordered = list(reversed(ids))
        else:
            ordered = list(ids)
        body = {
            "candidates": [
                {"id": cid, "readability": 10 - i, "coherent_clause": i == 0, "rationale": f"r{i}"}
                for i, cid in enumerate(ordered)
            ],
            "ranking": ordered,
            "best": ordered[0] if ordered else "",
        }
        return ModelResponse(
            content=[TextBlock(text=json.dumps(body))],
            usage=ModelUsage(input_tokens=120, output_tokens=50),
        )


@pytest.fixture
def reader_on(monkeypatch):
    monkeypatch.setenv("DECIPHER_FINALIST_READER", "llm:gpt-5.6-luna")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("DECIPHER_FINALIST_READER_SELECTS", raising=False)


def _use_provider(monkeypatch, provider):
    monkeypatch.setattr(llm_reader, "make_model_provider", lambda **_kw: provider)


# ---------------------------------------------------------------------------
# Strict env parsing
# ---------------------------------------------------------------------------


def test_finalist_reader_env_unset_is_off(monkeypatch):
    monkeypatch.delenv("DECIPHER_FINALIST_READER", raising=False)
    assert runner._parse_finalist_reader_spec() is None


@pytest.mark.parametrize("raw", ["garbage", "gpt-5.6-luna", "reader:foo", "llm:", "llm: "])
def test_finalist_reader_env_strict_raises_on_malformed(monkeypatch, raw):
    monkeypatch.setenv("DECIPHER_FINALIST_READER", raw)
    with pytest.raises(ValueError):
        runner._parse_finalist_reader_spec()


def test_finalist_reader_env_infers_and_explicit_provider(monkeypatch):
    monkeypatch.setenv("DECIPHER_FINALIST_READER", "llm:gpt-5.6-luna")
    spec = runner._parse_finalist_reader_spec()
    assert (spec.provider, spec.model) == ("openai", "gpt-5.6-luna")
    monkeypatch.setenv("DECIPHER_FINALIST_READER", "llm:openai:gpt-5.6-luna")
    spec = runner._parse_finalist_reader_spec()
    assert (spec.provider, spec.model) == ("openai", "gpt-5.6-luna")


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("", False), ("maybe", False)],
)
def test_finalist_reader_selects_strict_parse(monkeypatch, raw, expected):
    monkeypatch.setenv("DECIPHER_FINALIST_READER_SELECTS", raw)
    assert runner._finalist_reader_selects_enabled() is expected


def test_finalist_reader_selects_unset_is_false(monkeypatch):
    monkeypatch.delenv("DECIPHER_FINALIST_READER_SELECTS", raising=False)
    assert runner._finalist_reader_selects_enabled() is False


# ---------------------------------------------------------------------------
# Core reranker: annotate vs select
# ---------------------------------------------------------------------------


def _packets(n=3):
    return [
        {"packet": CandidatePacket(candidate_id=f"cand{i}", preview=f"KANDIDAT NUMMER {i} TEXT").to_dict()}
        for i in range(n)
    ]


def test_rank_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("DECIPHER_FINALIST_READER", raising=False)
    block, override = runner._finalist_reader_rank([{"preview": "x"}], language="de", allow_select=True)
    assert block is None and override is None


def test_rank_annotate_mode_records_block_no_override(monkeypatch, reader_on):
    provider = _CapturingProvider()
    _use_provider(monkeypatch, provider)
    packets = [p["packet"] for p in _packets(3)]
    block, override = runner._finalist_reader_rank(packets, language="de", allow_select=True)

    assert override is None  # selects is off -> annotate only
    assert block["status"] == "completed"
    assert block["mode"] == "annotate"
    assert block["selects_enabled"] is False
    assert block["selection_overridden"] is False
    assert block["model"] == "gpt-5.6-luna"
    assert block["provider"] == "openai"
    assert block["top_n"] == 3
    assert len(block["scores"]) == 3
    assert block["usage"]["input_tokens"] == 120
    assert block["usage"]["estimated_cost_usd"] > 0
    # reader preferred the last candidate, but selection is NOT changed.
    assert block["reader_best_candidate_id"] == "cand2"
    assert block["reader_best_index"] == 2


def test_rank_select_mode_returns_override(monkeypatch, reader_on):
    monkeypatch.setenv("DECIPHER_FINALIST_READER_SELECTS", "1")
    provider = _CapturingProvider()
    _use_provider(monkeypatch, provider)
    packets = [p["packet"] for p in _packets(3)]
    block, override = runner._finalist_reader_rank(packets, language="de", allow_select=True)

    assert override == 2
    assert block["mode"] == "select"
    assert block["selection_overridden"] is True
    assert block["overridden_from_index"] == 0
    assert block["overridden_to_index"] == 2


def test_rank_select_override_maps_past_textless_middle_candidate(monkeypatch, reader_on):
    """F2 regression: the reader drops textless candidates before scoring, so
    ``best_index`` indexes the PREPARED list. The override must map back to the
    ORIGINAL packet by candidate id, never install a dropped/textless row."""
    monkeypatch.setenv("DECIPHER_FINALIST_READER_SELECTS", "1")
    provider = _CapturingProvider()  # ranks the LAST display id first
    _use_provider(monkeypatch, provider)
    packets = [
        CandidatePacket(candidate_id="cand0", preview="KANDIDAT NULL TEXT HIER").to_dict(),
        CandidatePacket(candidate_id="cand1_textless", preview="").to_dict(),  # dropped in prep
        CandidatePacket(candidate_id="cand2", preview="KANDIDAT ZWEI TEXT HIER").to_dict(),
    ]
    block, override = runner._finalist_reader_rank(packets, language="de", allow_select=True)

    # Only cand0/cand2 survive prep -> reader sees C01/C02 and prefers C02
    # (cand2). Its PREPARED index is 1, but the override must resolve to the
    # ORIGINAL index 2 (cand2), NOT packets[1] (the textless cand1).
    assert override == 2
    assert packets[override]["candidate_id"] == "cand2"
    assert block["reader_best_candidate_id"] == "cand2"
    assert block["overridden_to_index"] == 2
    assert block["selection_overridden"] is True


def test_word_repair_route_is_annotate_only_even_with_selects(monkeypatch, reader_on):
    monkeypatch.setenv("DECIPHER_FINALIST_READER_SELECTS", "1")
    provider = _CapturingProvider()
    _use_provider(monkeypatch, provider)
    packets = [p["packet"] for p in _packets(3)]
    # allow_select=False models the word-repair route (null-mask route only).
    block, override = runner._finalist_reader_rank(packets, language="de", allow_select=False)
    assert override is None
    assert block["mode"] == "annotate"
    assert block["selection_overridden"] is False


def test_rank_graceful_no_key_degradation(monkeypatch):
    monkeypatch.setenv("DECIPHER_FINALIST_READER", "llm:gpt-5.6-luna")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_probe_api_key", lambda _p: "")
    # Provider must never be constructed when there is no key.
    _use_provider(monkeypatch, _CapturingProvider())
    packets = [p["packet"] for p in _packets(2)]
    block, override = runner._finalist_reader_rank(packets, language="de", allow_select=True)
    assert override is None
    assert block["status"] == "skipped"
    assert block["reason"] == "no_api_key"


def test_rank_malformed_env_never_crashes_records_error(monkeypatch):
    monkeypatch.setenv("DECIPHER_FINALIST_READER", "totally-bogus")
    block, override = runner._finalist_reader_rank([{"preview": "x"}], language="de", allow_select=True)
    assert override is None
    assert block["status"] == "error"
    assert "invalid_env" in block["reason"]


def test_runner_prompt_firewall_no_leak(monkeypatch, reader_on):
    """The prompt the runner constructs leaks no packet metadata."""
    provider = _CapturingProvider()
    _use_provider(monkeypatch, provider)
    packets = [
        CandidatePacket(
            candidate_id="mask:S17,S42",
            kind="null_mask",
            text="DIE ARBEIT DES LEHRLINGS",
            preview="DIE ARBEIT DES LEHRLINGS",
            solver_scores={"validation_score_v2": 0.418998},
            provenance={"mask": ["S17", "S42"], "key": {1: 2}},
            extras={"test_id": "copiale_single_B_copiale_p068"},
        ).to_dict(),
    ]
    runner._finalist_reader_rank(packets, language="de", allow_select=True)
    assert provider.sent_prompts, "reader was not invoked"
    llm_reader.assert_no_ground_truth_leak(
        provider.sent_prompts[0],
        forbidden=[
            "mask:S17,S42", "S17", "S42", "0.418998",
            "validation_score_v2", "copiale_single_B_copiale_p068", "test_id",
        ],
    )


# ---------------------------------------------------------------------------
# End-to-end: real null-mask bakeoff records the block / applies override
# ---------------------------------------------------------------------------


def _null_mask_bakeoff(monkeypatch):
    """Drive the real bakeoff with a tiny, stubbed homophonic solve."""
    symbols = ["S001", "S002", "S003", "S004"] * 24
    cipher_text = parse_canonical_transcription(" ".join(symbols))
    base_key = {0: 4, 1: 4, 2: 4, 3: 13}
    base_step = {
        "quality": {"collapsed": True, "top_letter_fraction": 0.30, "unique_letters": 8,
                    "letter_count": 96, "penalty": 0.0},
        "diagnostics": {"dict_rate": 0.2, "letter_count": 96, "segmentation_cost": 500},
        "anneal_score": -9.0,
        "selection_score": -9.0,
    }
    monkeypatch.setenv("DECIPHER_NULL_MASK_CANDIDATE_LIMIT", "4")
    monkeypatch.setenv("DECIPHER_NULL_MASK_MAX_SIZE", "1")
    monkeypatch.setenv("DECIPHER_NULL_MASK_MAX_MASKS", "2")
    monkeypatch.setenv("DECIPHER_NULL_MASK_TOP_N", "3")
    monkeypatch.setenv("DECIPHER_NULL_MASK_PROMOTE_TOP_N", "0")
    monkeypatch.setenv("DECIPHER_NULL_MASK_CONFIRM_TOP_N", "0")
    monkeypatch.setenv("DECIPHER_NULL_MASK_BEAM", "0")
    monkeypatch.setenv("DECIPHER_NULL_MASK_NEIGHBORHOOD", "0")

    def fake_run_homophonic(cipher_text, language, budget, refinement, solver_profile,
                            ground_truth, seed_offset=0, initial_key=None, fixed_cipher_ids=None):
        key = {0: 13, 1: 4, 2: 8, 3: 13}
        plaintext = "WENIGSICHUNDARBEIT" * 4
        step = {
            "quality": {"collapsed": False, "top_letter_fraction": 0.20, "unique_letters": 18,
                        "letter_count": len(plaintext), "penalty": 0.0},
            "diagnostics": {"dict_rate": 0.62, "letter_count": len(plaintext), "segmentation_cost": 80},
            "anneal_score": -4.0,
            "selection_score": -4.0,
        }
        return "zenith_native", key, plaintext, step

    monkeypatch.setattr(runner, "_run_homophonic", fake_run_homophonic)
    return runner._run_null_mask_bakeoff(
        cipher_text=cipher_text, language="de", budget="screen", solver_profile="zenith_native",
        base_solver="zenith_native", base_key=base_key, base_decryption="E" * len(symbols),
        base_step=base_step,
    )


def test_null_mask_bakeoff_no_reader_when_disabled(monkeypatch):
    monkeypatch.delenv("DECIPHER_FINALIST_READER", raising=False)
    report = _null_mask_bakeoff(monkeypatch)
    assert report["finalist_reader"] is None


def test_null_mask_bakeoff_annotate_leaves_selection_unchanged(monkeypatch, reader_on):
    provider = _CapturingProvider()
    _use_provider(monkeypatch, provider)
    report = _null_mask_bakeoff(monkeypatch)
    block = report["finalist_reader"]
    assert block is not None
    assert block["status"] == "completed"
    assert block["mode"] == "annotate"
    assert block["selection_overridden"] is False
    # The reader preferred the LAST finalist, but annotate mode must leave the
    # scalar winner (top finalist, index 0) installed: same candidate and mask.
    assert report["selected"]["candidate_id"] == report["top_finalists"][0]["candidate_id"]
    assert list(report["selected_mask"]) == list(report["top_finalists"][0].get("mask") or [])


def test_null_mask_bakeoff_selects_overrides_selection(monkeypatch, reader_on):
    monkeypatch.setenv("DECIPHER_FINALIST_READER_SELECTS", "1")
    provider = _CapturingProvider()
    _use_provider(monkeypatch, provider)
    report = _null_mask_bakeoff(monkeypatch)
    finalists = report["top_finalists"]
    block = report["finalist_reader"]
    assert block["selection_overridden"] is True
    # The reader preferred the LAST finalist; selection now points there.
    override_index = block["overridden_to_index"]
    assert override_index >= 1
    assert report["selected"]["candidate_id"] == finalists[override_index]["candidate_id"]
    assert list(report["selected_mask"]) == list(finalists[override_index].get("mask") or [])
    assert block["reader_selected_candidate_id"] == report["selected"]["candidate_id"]


# ---------------------------------------------------------------------------
# End-to-end: real word-repair refinement records the block (annotate-only)
# ---------------------------------------------------------------------------


def _small_cipher():
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText

    alpha = Alphabet(["S001", "S002", "S003"])
    tokens = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    raw = " ".join(alpha.symbol_for(t) for t in tokens)
    return CipherText(raw=raw, alphabet=alpha, separator=None)


def _wr_packet(page_validation, edits):
    return CandidatePacket(
        candidate_id="edits:" + ",".join(edits),
        kind="word_repair",
        source={"solver": "word_hypothesis_repair"},
        rank=1,
        text=None,
        preview="ARBEIT BRUDER",
        solver_scores={"page_validation_avg": page_validation, "adjudication_score": 1.0},
        validation={"accepted": True, "decision": "runtime_accept", "reasons": ["canned"]},
        provenance={"edits": edits, "mask": []},
    )


def test_word_repair_step_records_reader_block(monkeypatch, reader_on):
    import analysis.multipage as multipage
    import analysis.word_hypothesis_repair as whr

    monkeypatch.setattr(runner, "zenith_native_model_path", lambda _lang: None)
    monkeypatch.setattr(
        multipage, "score_page_runtime",
        lambda row, *, key, mask, language, model_path=None: {
            "validation_score_v2": 1.0, "validation_components_v2": {}, "language_quality_mean": 0.0,
            "language_quality_features": {}, "dict_rate": 0.0, "diagnostics": {}, "test_id": row["test_id"],
        },
    )
    monkeypatch.setattr(
        whr, "propose_word_repairs",
        lambda **_kw: [_wr_packet(2.5, ["S001:A->X"]), _wr_packet(2.0, ["S002:B->Y"])],
    )
    # selects is ON to prove word-repair still never overrides.
    monkeypatch.setenv("DECIPHER_FINALIST_READER_SELECTS", "1")
    provider = _CapturingProvider()
    _use_provider(monkeypatch, provider)

    step, _adopted = runner._run_word_repair_refinement(
        cipher_text=_small_cipher(), language="de", refinement="word_repair",
        base_solver="native_homophonic_anneal", base_key={0: 0, 1: 1, 2: 2},
        base_decryption="ABCABCABCABC", mask=(),
    )
    block = step["finalist_reader"]
    assert block is not None
    assert block["status"] == "completed"
    assert block["mode"] == "annotate"  # word-repair is annotate-only
    assert block["selection_overridden"] is False
    assert block["top_n"] == 2


def test_word_repair_step_no_reader_key_when_disabled(monkeypatch):
    import analysis.multipage as multipage
    import analysis.word_hypothesis_repair as whr

    monkeypatch.delenv("DECIPHER_FINALIST_READER", raising=False)
    monkeypatch.setattr(runner, "zenith_native_model_path", lambda _lang: None)
    monkeypatch.setattr(
        multipage, "score_page_runtime",
        lambda row, *, key, mask, language, model_path=None: {
            "validation_score_v2": 1.0, "validation_components_v2": {}, "language_quality_mean": 0.0,
            "language_quality_features": {}, "dict_rate": 0.0, "diagnostics": {}, "test_id": row["test_id"],
        },
    )
    monkeypatch.setattr(whr, "propose_word_repairs", lambda **_kw: [_wr_packet(2.5, ["S001:A->X"])])
    step, _adopted = runner._run_word_repair_refinement(
        cipher_text=_small_cipher(), language="de", refinement="word_repair",
        base_solver="native_homophonic_anneal", base_key={0: 0, 1: 1, 2: 2},
        base_decryption="ABCABCABCABC", mask=(),
    )
    assert "finalist_reader" not in step
