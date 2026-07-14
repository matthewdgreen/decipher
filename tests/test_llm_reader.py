"""Tests for the LLM reader library (improvement program Phase 4a, item 4.1).

The reader is exercised entirely with fake providers: structured parse,
retry-once then unscored fallback, provider-error resilience, budget caps, the
firewall-safe candidate text sourcing, and — the hard rule — the input firewall
(``reader_prompt_for`` leak-check).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.model_provider import ModelProviderError, ModelResponse, ModelUsage, TextBlock
from analysis.candidate_packet import CandidatePacket
from analysis.llm_reader import (
    DEFAULT_READER_MODEL,
    DEFAULT_READER_PROVIDER,
    LLMReaderConfig,
    ReaderResult,
    assert_no_ground_truth_leak,
    rank_candidates,
    reader_prompt_for,
)


class _FakeProvider:
    """A scripted model provider that records the prompts it is sent."""

    provider_name = "openai"

    def __init__(self, responses, *, model="gpt-5.6-luna"):
        self.model = model
        self._responses = list(responses)
        self.calls = 0
        self.sent_prompts: list[str] = []
        self.sent_systems: list[str] = []

    def send(self, *, messages, tools=None, system="", max_tokens=4096):
        self.calls += 1
        self.sent_prompts.append(messages[-1]["content"])
        self.sent_systems.append(system)
        if not self._responses:
            raise AssertionError("provider called more times than scripted")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        return ModelResponse(
            content=[TextBlock(text=text)],
            usage=ModelUsage(input_tokens=100, output_tokens=40),
        )


def _good_response(ids):
    return {
        "candidates": [
            {"id": cid, "readability": 10 - i, "coherent_clause": i == 0, "rationale": f"r{i}"}
            for i, cid in enumerate(ids)
        ],
        "ranking": list(ids),
        "best": ids[0],
    }


def test_structured_parse_maps_back_to_original_ids():
    cands = [
        {"candidate_id": "mask:S1,S2", "preview": "DIE ARBEIT DES LEHRLINGS"},
        {"candidate_id": "mask:S3", "preview": "XQZ NONSENSE FRAGMENT"},
    ]
    # The model addresses candidates by the neutral display ids C01/C02.
    provider = _FakeProvider([_good_response(["C02", "C01"])])
    cfg = LLMReaderConfig()
    result = rank_candidates(cands, cfg, provider=provider)

    assert result.parse_ok is True
    assert result.error is None
    # Ranking and best are mapped back to the caller's original ids.
    assert result.ranking == ["mask:S3", "mask:S1,S2"]
    assert result.best_candidate_id == "mask:S3"
    assert result.best_index == 1
    by_id = {s.candidate_id: s for s in result.scores}
    assert by_id["mask:S3"].readability == 10.0
    assert by_id["mask:S3"].coherent_clause is True
    assert by_id["mask:S3"].rank == 1
    assert by_id["mask:S1,S2"].rank == 2
    assert result.usage["input_tokens"] == 100
    assert result.usage["estimated_cost_usd"] > 0
    assert result.usage["calls"] == 1


def test_retry_once_then_success():
    cands = [{"candidate_id": "A", "preview": "DIE ARBEIT"}, {"candidate_id": "B", "preview": "NONSENSE"}]
    # First response is unparsable, second is valid -> exactly one retry.
    provider = _FakeProvider(["not json at all", _good_response(["C01", "C02"])])
    result = rank_candidates(cands, LLMReaderConfig(max_calls=2), provider=provider)

    assert provider.calls == 2
    assert result.parse_ok is True
    assert result.calls == 2
    assert result.best_candidate_id == "A"
    # Usage accumulates across both calls.
    assert result.usage["input_tokens"] == 200


def test_retry_exhausted_marks_all_unscored():
    cands = [{"candidate_id": "A", "preview": "DIE ARBEIT"}, {"candidate_id": "B", "preview": "NONSENSE"}]
    provider = _FakeProvider(["garbage one", "garbage two"])
    result = rank_candidates(cands, LLMReaderConfig(max_calls=2), provider=provider)

    assert provider.calls == 2
    assert result.parse_ok is False
    assert result.error == "parse_failed"
    assert result.ranking == []
    assert result.best_candidate_id is None
    assert set(result.unscored) == {"A", "B"}
    assert all(s.unscored for s in result.scores)
    # Even on total failure the caller gets usage, never an exception.
    assert result.usage["calls"] == 2


def test_provider_error_never_crashes():
    cands = [{"candidate_id": "A", "preview": "DIE ARBEIT"}]
    provider = _FakeProvider([ModelProviderError("overloaded")])
    result = rank_candidates(cands, LLMReaderConfig(), provider=provider)

    assert result.parse_ok is False
    assert result.error is not None and "overloaded" in result.error
    assert all(s.unscored for s in result.scores)


def test_no_candidates_returns_error_not_crash():
    result = rank_candidates([{"candidate_id": "A", "preview": ""}], LLMReaderConfig(), provider=_FakeProvider([]))
    assert result.error == "no_candidates"
    assert result.scores == []


def test_budget_caps_truncate_visibly_in_prompt():
    cands = [
        {"candidate_id": f"c{i}", "preview": ("WORT " * 400) + f"END{i}"}
        for i in range(20)
    ]
    cfg = LLMReaderConfig(max_candidates=3, max_chars_per_candidate=50)
    prompt = reader_prompt_for(cands, cfg)

    # max_candidates: only C01..C03 appear, C04 does not.
    assert "C03:" in prompt
    assert "C04:" not in prompt
    # max_chars_per_candidate: the 2000-char preview is sliced to 50 chars, so
    # the trailing "END0" marker never reaches the prompt.
    assert "END0" not in prompt
    # Each candidate block's text is at most 50 chars.
    body = prompt.split("Candidate texts:\n\n", 1)[1]
    for block in body.split("\n\n"):
        _label, _, text = block.partition("\n")
        if text:
            assert len(text) <= 50


def test_word_repair_packet_text_sourcing_uses_preview_and_page_previews():
    # Word-repair packets carry page_previews as a LIST OF DICTS
    # ({"test_id", "preview", "filtered_length"}) — the real runtime shape. Only
    # the preview text is firewall-safe; the test_id/filtered_length must not leak.
    packet = CandidatePacket(
        candidate_id="edits:S3:A->B",
        kind="word_repair",
        text=None,
        preview="DIE ARBEIT DES LEHRLINGS",
        provenance={"edits": ["S3:A->B"], "key": {1: 2, 3: 4}},
        extras={
            "page_previews": [
                {
                    "test_id": "copiale_single_B_copiale_p068",
                    "preview": "UND DER BRUDER SPRACH",
                    "filtered_length": 21,
                },
                {
                    "test_id": "copiale_single_B_copiale_p069",
                    "preview": "JEDER SOLL DIE HAND HEBEN",
                    "filtered_length": 25,
                },
            ]
        },
    )
    prompt = reader_prompt_for([packet], LLMReaderConfig())
    assert "DIE ARBEIT DES LEHRLINGS" in prompt
    assert "UND DER BRUDER SPRACH" in prompt
    assert "JEDER SOLL DIE HAND HEBEN" in prompt
    # Regression pin: only entry["preview"] is sourced — never the dict itself.
    assert_no_ground_truth_leak(
        prompt,
        forbidden=[
            "copiale_single_B_copiale_p068",
            "copiale_single_B_copiale_p069",
            "test_id",
            "filtered_length",
        ],
    )


def test_prompt_firewall_no_metadata_leak():
    """HARD RULE: only neutral ids + text excerpts + language reach the model."""
    packets = [
        CandidatePacket(
            candidate_id="mask:S17,S42",  # mask provenance must not leak
            kind="null_mask",
            text="DIE ARBEIT DES LEHRLINGS WIRD GEPRUEFT",
            preview="DIE ARBEIT DES LEHRLINGS",
            solver_scores={"validation_score_v2": 0.418998, "selection_score": -7.303010},
            provenance={"mask": ["S17", "S42"], "key": {1: 2, 3: 4}},
            extras={"post_hoc_char_avg": 0.5985915, "test_id": "copiale_single_B_copiale_p068"},
        ),
        CandidatePacket(
            candidate_id="edits:S3:X->Y,S9:Q->Z",  # edit provenance must not leak
            kind="word_repair",
            text=None,
            preview="JEDER BRUDER SOLL DIE HAND HEBEN",
            solver_scores={"adjudication_score": -3.0},
            provenance={"edits": ["S3:X->Y", "S9:Q->Z"], "key": {5: 6}},
            # Real word-repair shape: page_previews is a list of dicts. The
            # test_id inside must never reach the prompt (only ["preview"] does).
            extras={
                "page_previews": [
                    {
                        "test_id": "copiale_single_B_copiale_p070",
                        "preview": "DER MEISTER GAB DAS ZEICHEN",
                        "filtered_length": 27,
                    },
                ]
            },
        ),
    ]
    cfg = LLMReaderConfig(language="de")
    prompt = reader_prompt_for(packets, cfg)

    # The exact prompt leak-checks clean against every forbidden metadatum.
    assert_no_ground_truth_leak(
        prompt,
        forbidden=[
            "mask:S17,S42",
            "S17",
            "S42",
            "edits:S3:X->Y,S9:Q->Z",
            "S3:X->Y",
            "0.418998",
            "-7.303010",
            "validation_score_v2",
            "selection_score",
            "adjudication_score",
            "post_hoc_char_avg",
            "copiale_single_B_copiale_p068",
            "copiale_single_B_copiale_p070",  # test_id inside word-repair page_previews dict
            "filtered_length",
            "test_id",
        ],
    )
    # Positive control: the candidate text (allowed) IS present — including the
    # word-repair page preview, sourced from the dict's "preview" field only.
    assert "DIE ARBEIT DES LEHRLINGS" in prompt
    assert "DER MEISTER GAB DAS ZEICHEN" in prompt
    with pytest.raises(AssertionError):
        assert_no_ground_truth_leak(prompt, forbidden=["DIE ARBEIT DES LEHRLINGS"])


def test_language_name_appears_in_prompt():
    prompt = reader_prompt_for([{"candidate_id": "A", "preview": "HELLO WORLD"}], LLMReaderConfig(language="de"))
    assert "German" in prompt
    prompt_la = reader_prompt_for([{"candidate_id": "A", "preview": "LOREM"}], LLMReaderConfig(language="la"))
    assert "Latin" in prompt_la


def test_defaults_are_luna():
    cfg = LLMReaderConfig()
    assert cfg.provider == DEFAULT_READER_PROVIDER == "openai"
    assert cfg.model == DEFAULT_READER_MODEL == "gpt-5.6-luna"
    assert cfg.max_candidates == 12
    assert cfg.max_chars_per_candidate == 700


def test_partial_id_coverage_still_scores_present_ids():
    cands = [{"candidate_id": "A", "preview": "DIE ARBEIT"}, {"candidate_id": "B", "preview": "NONSENSE"}]
    # Model only addresses C01; C02 must come back unscored, not crash.
    response = {
        "candidates": [{"id": "C01", "readability": 9, "coherent_clause": True, "rationale": "ok"}],
        "ranking": ["C01"],
        "best": "C01",
    }
    result = rank_candidates(cands, LLMReaderConfig(), provider=_FakeProvider([response]))
    # Half-coverage of two candidates (one addressed) meets the validity floor.
    assert result.parse_ok is True
    by_id = {s.candidate_id: s for s in result.scores}
    assert by_id["A"].readability == 9.0
    assert by_id["B"].unscored is True
    assert "B" in result.unscored
