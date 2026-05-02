from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis import ngram
from analysis.polyalphabetic import (
    encode_quagmire_plaintext,
    search_keyed_vigenere_alphabet_anneal,
    search_quagmire3_keyword_alphabet,
)
from analysis.polyalphabetic_fast import (
    FAST_AVAILABLE,
    estimate_quagmire3_shotgun_budget,
    normalized_ngram_score_fast,
    search_quagmire3_shotgun_fast,
    search_keyed_vigenere_alphabet_anneal_fast,
)
from analysis.transform_fast import score_transform_candidates_fast_batch
from analysis.transform_search import (
    _fast_structural_metrics_for_order,
    iter_transform_candidates,
    screen_transform_candidates,
    validate_transform_candidate,
)
from agent.tools_v2 import WorkspaceToolExecutor
from automated.runner import run_automated
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


PLAINTEXT = "THEOLDMANANDTHESEAWASWRITTENBYERNESTHEMINGWAY"
KEY = "LEMON"


def _vigenere_encrypt(plaintext: str, key: str) -> str:
    shifts = [ord(ch) - ord("A") for ch in key]
    out = []
    for i, ch in enumerate(plaintext):
        out.append(chr((ord(ch) - ord("A") + shifts[i % len(shifts)]) % 26 + ord("A")))
    return "".join(out)


def _cipher_text() -> CipherText:
    ciphertext = _vigenere_encrypt(PLAINTEXT, KEY)
    return CipherText(raw=ciphertext, alphabet=Alphabet.from_text(ciphertext), separator=None)


pytestmark = pytest.mark.skipif(
    not FAST_AVAILABLE,
    reason="optional decipher_fast Rust extension is not installed",
)


def test_fast_normalized_ngram_score_matches_python():
    quad = ngram.NGRAM_CACHE.get("en", 4)
    text = "THE QUICK BROWN FOX JUMPS"
    assert normalized_ngram_score_fast(text, quad, 4) == pytest.approx(
        ngram.normalized_ngram_score(text, quad, n=4),
        abs=1e-12,
    )


def test_fast_transform_structural_batch_matches_python_reference():
    tokens = list(range(88))
    candidates = list(
        iter_transform_candidates(
            token_count=len(tokens),
            columns=11,
            profile="medium",
            max_candidates=200,
        )
    )

    results = score_transform_candidates_fast_batch(
        tokens,
        [candidate.to_dict() for candidate in candidates],
        threads=2,
    )

    assert len(results) == len(candidates)
    for candidate, result in zip(candidates, results, strict=True):
        validation = validate_transform_candidate(len(tokens), candidate)
        assert result["valid"] == validation["valid"]
        if not validation["valid"]:
            assert result["reason"] == validation["reason"]
            continue
        assert result["token_order_hash"] == validation["token_order_hash"]
        assert result["position_order_preview"] == validation["position_order"][:60]
        expected_metrics = _fast_structural_metrics_for_order(
            validation["position_order"],
            candidate.grid,
            None,
        )
        for key, expected in expected_metrics.items():
            assert result["metrics"][key] == pytest.approx(expected, abs=1e-12)


def test_wide_transform_screen_uses_rust_structural_kernel(monkeypatch):
    monkeypatch.setenv("DECIPHER_TRANSFORM_SCREEN_RUST", "1")
    screen = screen_transform_candidates(
        list(range(88)),
        columns=11,
        profile="wide",
        top_n=5,
        max_generated_candidates=100_000,
        streaming=True,
        include_program_search=False,
    )

    assert screen["streaming"] is True
    assert screen["fast_structural_metrics"] is True
    assert screen["rust_structural_metrics"] is True
    assert screen["candidate_count"] > 0
    assert screen["deduped_candidate_count"] > 0
    assert screen["top_candidates"]


def test_fast_keyed_vigenere_steps_zero_matches_python_baseline():
    ct = _cipher_text()
    py_result = search_keyed_vigenere_alphabet_anneal(
        ct,
        language="en",
        max_period=8,
        include_standard_alphabet=True,
        steps=0,
        restarts=1,
        top_n=1,
    )
    fast_result = search_keyed_vigenere_alphabet_anneal_fast(
        ct,
        language="en",
        max_period=8,
        include_standard_alphabet=True,
        steps=0,
        restarts=1,
        top_n=1,
    )
    assert fast_result["best_candidate"]["plaintext"] == py_result["best_candidate"]["plaintext"]
    assert fast_result["best_candidate"]["score"] == pytest.approx(
        py_result["best_candidate"]["score"],
        abs=1e-5,
    )


def test_fast_quagmire3_seeded_search_recovers_known_candidate():
    plaintext = (
        "ITWASTOTALLYINVISIBLEHOWSTHATPOSSIBLETHEYUSEDTHEEARTHS"
        "MAGNETICFIELDXTHEINFORMATIONWASGATHEREDANDTRANSMITTED"
        "UNDERGRUUNDTOANUNKNOWNLOCATIONXDOESLANGLEYKNOWABOUTTHIS"
        "THEYSHOULDITSBURIEDOUTTHERESOMEWHEREXWHOKNOWSTHEEXACT"
        "LOCATIONONLYWWTHISWASHISLASTMESSAGEXTHIRTYEIGHTDEGREES"
        "FIFTYSEVENMINUTESSIXPOINTFIVESECONDSNORTHSEVENTYSEVEN"
        "DEGREESEIGHTMINUTESFORTYFOURSECONDSWESTXLAYERTWO"
    )
    ciphertext = encode_quagmire_plaintext(
        plaintext,
        cycleword="ABSCISSA",
        quagmire_type="quag3",
        alphabet_keyword="KRYPTOS",
    )
    ct = CipherText(raw=ciphertext, alphabet=Alphabet.from_text(ciphertext), separator=None)

    result = search_quagmire3_shotgun_fast(
        ct,
        language="en",
        keyword_lengths=[7],
        cycleword_lengths=[8],
        initial_keywords=["KRYPTOS"],
        hillclimbs=0,
        restarts=1,
        threads=1,
        top_n=1,
    )

    best = result["best_candidate"]
    assert result["solver"] == "quagmire3_shotgun_rust"
    assert result["threads"] == 1
    assert best["metadata"]["alphabet_keyword"] == "KRYPTOS"
    assert best["metadata"]["cycleword"] == "ABSCISSA"
    assert best["plaintext"] == plaintext


def test_quagmire3_shotgun_budget_estimate_sizes_candidate_count():
    estimate = estimate_quagmire3_shotgun_budget(
        keyword_lengths=[7, 8],
        cycleword_lengths=[8],
        hillclimbs=10_000,
        restarts=500,
        threads=4,
    )

    assert estimate["restart_jobs"] == 1000
    assert estimate["nominal_proposals"] == 10_000_000
    assert estimate["estimated_threads"] == 4
    assert estimate["estimated_minutes_typical"] > 0
    assert "diagnostic" in estimate["sizing_guidance"][0]


def test_fast_quagmire3_seeded_zero_step_matches_python_semantics():
    plaintext = (
        "ITWASTOTALLYINVISIBLEHOWSTHATPOSSIBLETHEYUSEDTHEEARTHS"
        "MAGNETICFIELDXTHEINFORMATIONWASGATHEREDANDTRANSMITTED"
        "UNDERGRUUNDTOANUNKNOWNLOCATIONXDOESLANGLEYKNOWABOUTTHIS"
        "THEYSHOULDITSBURIEDOUTTHERESOMEWHEREXWHOKNOWSTHEEXACT"
        "LOCATIONONLYWWTHISWASHISLASTMESSAGEXTHIRTYEIGHTDEGREES"
        "FIFTYSEVENMINUTESSIXPOINTFIVESECONDSNORTHSEVENTYSEVEN"
        "DEGREESEIGHTMINUTESFORTYFOURSECONDSWESTXLAYERTWO"
    )
    ciphertext = encode_quagmire_plaintext(
        plaintext,
        cycleword="ABSCISSA",
        quagmire_type="quag3",
        alphabet_keyword="KRYPTOS",
    )
    ct = CipherText(raw=ciphertext, alphabet=Alphabet.from_text(ciphertext), separator=None)

    py_result = search_quagmire3_keyword_alphabet(
        ct,
        language="en",
        keyword_lengths=[7],
        cycleword_lengths=[8],
        initial_keywords=["KRYPTOS"],
        steps=0,
        restarts=1,
        top_n=1,
    )
    fast_result = search_quagmire3_shotgun_fast(
        ct,
        language="en",
        keyword_lengths=[7],
        cycleword_lengths=[8],
        initial_keywords=["KRYPTOS"],
        hillclimbs=0,
        restarts=1,
        threads=1,
        top_n=1,
    )

    py_best = py_result["best_candidate"]
    fast_best = fast_result["best_candidate"]
    assert py_best["metadata"]["alphabet_keyword"] == fast_best["metadata"]["alphabet_keyword"]
    assert py_best["metadata"]["cycleword"] == fast_best["metadata"]["cycleword"]
    assert py_best["plaintext"] == fast_best["plaintext"] == plaintext


def test_agent_quagmire_tool_can_use_rust_shotgun_engine():
    plaintext = (
        "ITWASTOTALLYINVISIBLEHOWSTHATPOSSIBLETHEYUSEDTHEEARTHS"
        "MAGNETICFIELDXTHEINFORMATIONWASGATHEREDANDTRANSMITTED"
        "UNDERGRUUNDTOANUNKNOWNLOCATIONXDOESLANGLEYKNOWABOUTTHIS"
        "THEYSHOULDITSBURIEDOUTTHERESOMEWHEREXWHOKNOWSTHEEXACT"
        "LOCATIONONLYWWTHISWASHISLASTMESSAGEXTHIRTYEIGHTDEGREES"
        "FIFTYSEVENMINUTESSIXPOINTFIVESECONDSNORTHSEVENTYSEVEN"
        "DEGREESEIGHTMINUTESFORTYFOURSECONDSWESTXLAYERTWO"
    )
    ciphertext = encode_quagmire_plaintext(
        plaintext,
        cycleword="ABSCISSA",
        quagmire_type="quag3",
        alphabet_keyword="KRYPTOS",
    )
    ct = CipherText(raw=ciphertext, alphabet=Alphabet.standard_english(), separator=None)
    executor = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "INFORMATION", "LOCATION", "SECONDS", "DEGREES"},
        word_list=["THE", "INFORMATION", "LOCATION", "SECONDS", "DEGREES"],
        pattern_dict={},
    )

    result = executor._tool_search_quagmire3_keyword_alphabet({
        "branch": "main",
        "engine": "rust_shotgun",
        "keyword_lengths": [7],
        "cycleword_lengths": [8],
        "initial_keywords": ["KRYPTOS"],
        "hillclimbs": 0,
        "restarts": 1,
        "threads": 1,
        "top_n": 1,
        "install_top_n": 1,
        "new_branch_prefix": "quag_rust",
    })

    installed = result["installed_branches"][0]
    branch = executor.workspace.get_branch(installed["branch"])
    assert result["status"] == "completed"
    assert result["solver"] == "quagmire3_shotgun_rust"
    assert result["threads"] == 1
    assert result["rust_fast_kernel"]["available"] is True
    assert "not an equivalent runtime path" in result["engine_equivalence_note"]
    assert branch.metadata["search_metadata"]["engine"] == "rust_shotgun"
    assert branch.metadata["search_metadata"]["hillclimbs_per_restart"] == 0
    assert branch.metadata["alphabet_keyword"] == "KRYPTOS"
    assert branch.metadata["cycleword"] == "ABSCISSA"
    assert branch.metadata["decoded_text"] == plaintext


def test_agent_quagmire_tool_estimate_only_does_not_install_branch():
    ct = CipherText(raw="VFPJUDEEHZWETZYVGWHKKQETGFQJNCEGGWHKK", alphabet=Alphabet.standard_english(), separator=None)
    executor = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set=set(),
        word_list=[],
        pattern_dict={},
    )

    result = executor._tool_search_quagmire3_keyword_alphabet({
        "branch": "main",
        "engine": "rust_shotgun",
        "keyword_lengths": [7],
        "cycleword_lengths": [8],
        "hillclimbs": 10_000,
        "restarts": 500,
        "threads": 4,
        "estimate_only": True,
    })

    assert result["status"] == "estimated"
    assert result["installed_branches"] == []
    assert result["budget_estimate"]["nominal_proposals"] == 5_000_000
    assert result["budget_estimate"]["estimated_threads"] == 4
    assert "rerun this tool" in result["recommended_next_action"]


def test_automated_quagmire_search_can_use_rust_shotgun_engine(monkeypatch):
    plaintext = (
        "ITWASTOTALLYINVISIBLEHOWSTHATPOSSIBLETHEYUSEDTHEEARTHS"
        "MAGNETICFIELDXTHEINFORMATIONWASGATHEREDANDTRANSMITTED"
        "UNDERGRUUNDTOANUNKNOWNLOCATIONXDOESLANGLEYKNOWABOUTTHIS"
        "THEYSHOULDITSBURIEDOUTTHERESOMEWHEREXWHOKNOWSTHEEXACT"
        "LOCATIONONLYWWTHISWASHISLASTMESSAGEXTHIRTYEIGHTDEGREES"
        "FIFTYSEVENMINUTESSIXPOINTFIVESECONDSNORTHSEVENTYSEVEN"
        "DEGREESEIGHTMINUTESFORTYFOURSECONDSWESTXLAYERTWO"
    )
    ciphertext = encode_quagmire_plaintext(
        plaintext,
        cycleword="ABSCISSA",
        quagmire_type="quag3",
        alphabet_keyword="KRYPTOS",
    )
    ct = CipherText(raw=ciphertext, alphabet=Alphabet.from_text(ciphertext), separator=None)
    monkeypatch.setenv("DECIPHER_KEYED_VIGENERE_MODE", "quagmire_search")
    monkeypatch.setenv("DECIPHER_QUAGMIRE_ENGINE", "rust_shotgun")
    monkeypatch.setenv("DECIPHER_QUAGMIRE_INITIAL_KEYWORDS", "KRYPTOS")
    monkeypatch.setenv("DECIPHER_QUAGMIRE_KEYWORD_LENGTHS", "7")
    monkeypatch.setenv("DECIPHER_QUAGMIRE_CYCLEWORD_LENGTHS", "8")
    monkeypatch.setenv("DECIPHER_QUAGMIRE_HILLCLIMBS", "0")
    monkeypatch.setenv("DECIPHER_QUAGMIRE_SEARCH_RESTARTS", "1")
    monkeypatch.setenv("DECIPHER_QUAGMIRE_THREADS", "1")

    result = run_automated(
        ct,
        language="en",
        cipher_id="k2_quagmire_rust_search",
        ground_truth=plaintext,
        cipher_system="quagmire3",
    )

    assert result.status == "completed"
    assert result.solver == "quagmire3_shotgun_rust"
    assert result.final_decryption == plaintext
    step = next(step for step in result.steps if step["name"] == "search_quagmire3_keyword_alphabet")
    assert step["engine"] == "rust_shotgun"
    assert step["threads"] == 1
    assert step["hillclimbs_per_restart"] == 0
    assert step["restart_jobs"] == 1
    assert step["cycleword"] == "ABSCISSA"
    assert step["alphabet_keyword"] == "KRYPTOS"
