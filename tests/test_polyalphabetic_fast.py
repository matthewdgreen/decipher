from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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
from analysis.pure_transposition import (
    clear_pure_transposition_cache,
    generate_pure_transposition_candidates,
    screen_pure_transposition,
)
from analysis.transform_fast import score_transform_candidates_fast_batch, search_k3_transmatrix_fast
from analysis.transform_search import (
    _fast_structural_metrics_for_order,
    iter_transform_candidates,
    screen_transform_candidates,
    validate_transform_candidate,
)
from analysis.transformers import TransformPipeline, TransformStep, apply_transform_pipeline
from agent.tools_v2 import WorkspaceToolExecutor
from benchmark.loader import parse_canonical_transcription
from automated.runner import run_automated
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from testgen.builder import build_test_case
from testgen.cache import PlaintextCache
from testgen.spec import TestSpec as SyntheticTestSpec
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


def test_fast_matrix_rotate_transform_matches_blake_semantics():
    tokens = list(range(12))
    results = score_transform_candidates_fast_batch(
        tokens,
        [
            {
                "candidate_id": "matrix_rotate_width4_cw",
                "pipeline": {
                    "steps": [
                        {
                            "name": "MatrixRotate",
                            "data": {"width": 4, "direction": "cw"},
                        }
                    ]
                },
            }
        ],
        threads=1,
    )

    assert results[0]["valid"] is True
    assert results[0]["position_order_preview"] == [8, 4, 0, 9, 5, 1, 10, 6, 2, 11, 7, 3]
    assert results[0]["transformed_preview"] == [8, 4, 0, 9, 5, 1, 10, 6, 2, 11, 7, 3]


def test_python_matrix_rotate_and_transmatrix_match_fast_semantics():
    tokens = list(range(12))
    matrix_pipeline = TransformPipeline(
        steps=(TransformStep("MatrixRotate", {"width": 4, "direction": "cw"}),)
    )
    assert apply_transform_pipeline(tokens, matrix_pipeline).tokens == [
        8,
        4,
        0,
        9,
        5,
        1,
        10,
        6,
        2,
        11,
        7,
        3,
    ]

    transmatrix_pipeline = TransformPipeline(
        steps=(TransformStep("TransMatrix", {"w1": 4, "w2": 3, "direction": "cw"}),)
    )
    expected = apply_transform_pipeline(
        apply_transform_pipeline(tokens, matrix_pipeline).tokens,
        TransformPipeline(steps=(TransformStep("MatrixRotate", {"width": 3, "direction": "cw"}),)),
    ).tokens
    assert apply_transform_pipeline(tokens, transmatrix_pipeline).tokens == expected


def test_pure_transposition_candidates_include_matrix_rotate_breadth():
    candidates = generate_pure_transposition_candidates(
        token_count=120,
        profile="medium",
        include_transmatrix=True,
        transmatrix_max_width=20,
    )

    families = {candidate.family for candidate in candidates}
    assert "matrix_rotate_cw" in families
    assert "matrix_rotate_ccw" in families
    assert "transmatrix" in families


def test_fast_k3_transmatrix_search_recovers_known_k3_plaintext():
    repo_root = Path(__file__).resolve().parents[1]
    k3_path = (
        repo_root
        / "other_tools"
        / "zenith-src"
        / "zenith-inference"
        / "src"
        / "main"
        / "resources"
        / "ciphers"
        / "kryptos3.json"
    )
    payload = json.loads(k3_path.read_text(encoding="utf-8"))
    values = [ord(ch) - ord("A") for ch in payload["ciphertext"] if "A" <= ch <= "Z"]

    result = search_k3_transmatrix_fast(
        values,
        language="en",
        min_width=2,
        max_width=60,
        top_n=5,
        threads=2,
    )

    best = result["top_candidates"][0]
    assert result["solver"] == "k3_transmatrix_rust"
    assert result["candidate_count"] == (60 - 2 + 1) ** 2 * 2
    assert best["variant"] == "k3_transmatrix"
    assert best["plaintext"].startswith("SLOWLYDESPARATLYSLOWLYTHEREMAINS")
    assert best["pipeline"]["steps"][0]["name"] == "TransMatrix"


def test_pure_transposition_screen_recovers_k3_transmatrix():
    repo_root = Path(__file__).resolve().parents[1]
    k3_path = (
        repo_root
        / "other_tools"
        / "zenith-src"
        / "zenith-inference"
        / "src"
        / "main"
        / "resources"
        / "ciphers"
        / "kryptos3.json"
    )
    payload = json.loads(k3_path.read_text(encoding="utf-8"))
    ciphertext = "".join(ch for ch in payload["ciphertext"] if "A" <= ch <= "Z")
    ct = CipherText(raw=ciphertext, alphabet=Alphabet.from_text(ciphertext), separator=None)

    result = screen_pure_transposition(
        ct,
        language="en",
        profile="wide",
        top_n=5,
        transmatrix_max_width=60,
        threads=2,
    )

    best = result["best_candidate"]
    assert result["solver"] == "pure_transposition_screen_rust"
    assert best["family"] == "transmatrix"
    assert best["plaintext"].startswith("SLOWLYDESPARATLYSLOWLYTHEREMAINS")


def test_pure_transposition_screen_recovers_non_k3_route_candidate():
    base_plaintext = (
        "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGANDTHENRUNSACROSSTHE"
        "FIELDWHILEWATCHINGTHEMOONRISEABOVETHEQUIETRIVERBANK"
    )
    plaintext = (base_plaintext * 2)[:120]
    cipher = _inverse_pipeline_cipher(
        plaintext,
        TransformPipeline(
            steps=(TransformStep("RouteRead", {"route": "columns_down"}),),
            columns=20,
            rows=6,
        ),
    )
    ct = CipherText(raw=cipher, alphabet=Alphabet.standard_english(), separator=None)

    result = screen_pure_transposition(
        ct,
        language="en",
        profile="small",
        top_n=10,
        include_transmatrix=False,
        threads=2,
    )

    best = result["best_candidate"]
    assert best["family"] == "route_columns_down"
    assert best["plaintext"] == plaintext


def test_pure_transposition_screen_recovers_matrix_rotate_candidate():
    base_plaintext = (
        "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGANDTHENRUNSACROSSTHE"
        "FIELDWHILEWATCHINGTHEMOONRISEABOVETHEQUIETRIVERBANK"
    )
    plaintext = (base_plaintext * 2)[:120]
    cipher = _inverse_pipeline_cipher(
        plaintext,
        TransformPipeline(
            steps=(TransformStep("MatrixRotate", {"width": 17, "direction": "cw"}),),
        ),
    )
    ct = CipherText(raw=cipher, alphabet=Alphabet.standard_english(), separator=None)

    result = screen_pure_transposition(
        ct,
        language="en",
        profile="medium",
        top_n=10,
        include_transmatrix=False,
        threads=2,
    )

    matrix_hits = [
        candidate
        for candidate in result["top_candidates"]
        if candidate["family"] == "matrix_rotate_cw"
        and candidate["params"]["width"] == 17
        and candidate["plaintext"] == plaintext
    ]
    assert matrix_hits
    assert result["candidate_plan"]["include_matrix_rotate"] is True


def test_synthetic_builder_can_generate_pure_transposition_case(tmp_path):
    spec = SyntheticTestSpec(
        language="en",
        approx_length=25,
        word_boundaries=False,
        transposition_only=True,
        seed=41,
        transform_pipeline={
            "steps": [{"name": "MatrixRotate", "data": {"width": 7, "direction": "cw"}}],
        },
    )
    cache = PlaintextCache(tmp_path / "cache")
    cache.put(spec, "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG")

    test_data = build_test_case(spec, cache, api_key="", seed=41)
    ct = parse_canonical_transcription(test_data.canonical_transcription)
    result = screen_pure_transposition(
        ct,
        language="en",
        profile="small",
        top_n=5,
        include_transmatrix=False,
        threads=2,
    )

    assert test_data.test.cipher_system == "pure_transposition"
    assert test_data.test.test_id == "synth_en_25ptnb_s41"
    assert test_data.plaintext == "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
    assert result["best_candidate"]["plaintext"] == test_data.plaintext


def test_pure_transposition_screen_cache_reuses_identical_search():
    clear_pure_transposition_cache()
    plaintext = (
        "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGANDTHENRUNSACROSSTHE"
        "FIELDWHILEWATCHINGTHEMOONRISEABOVETHEQUIETRIVERBANK"
    )
    plaintext = (plaintext * 2)[:120]
    cipher = _inverse_pipeline_cipher(
        plaintext,
        TransformPipeline(
            steps=(TransformStep("RouteRead", {"route": "columns_down"}),),
            columns=20,
            rows=6,
        ),
    )
    ct = CipherText(raw=cipher, alphabet=Alphabet.standard_english(), separator=None)

    first = screen_pure_transposition(
        ct,
        language="en",
        profile="small",
        top_n=5,
        include_transmatrix=False,
        threads=2,
    )
    second = screen_pure_transposition(
        ct,
        language="en",
        profile="small",
        top_n=5,
        include_transmatrix=False,
        threads=2,
    )

    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is True
    assert second["best_candidate"]["plaintext"] == plaintext
    assert second["best_candidate"]["candidate_id"] == first["best_candidate"]["candidate_id"]


def test_agent_pure_transposition_tool_installs_readable_branch():
    base_plaintext = (
        "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGANDTHENRUNSACROSSTHE"
        "FIELDWHILEWATCHINGTHEMOONRISEABOVETHEQUIETRIVERBANK"
    )
    plaintext = (base_plaintext * 2)[:120]
    cipher = _inverse_pipeline_cipher(
        plaintext,
        TransformPipeline(
            steps=(TransformStep("RouteRead", {"route": "columns_down"}),),
            columns=20,
            rows=6,
        ),
    )
    ct = CipherText(raw=cipher, alphabet=Alphabet.standard_english(), separator=None)
    executor = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "QUICK", "BROWN", "RIVER"},
        word_list=["THE", "QUICK", "BROWN", "RIVER"],
        pattern_dict={},
    )

    result = executor._tool_search_pure_transposition({
        "branch": "main",
        "profile": "small",
        "include_transmatrix": False,
        "top_n": 5,
        "install_top_n": 1,
        "threads": 2,
    })

    installed = result["installed_branches"][0]
    branch = executor.workspace.get_branch(installed["branch"])
    assert result["status"] == "completed"
    assert result["solver"] == "pure_transposition_screen_rust"
    assert result["search_session_id"].startswith("pure_transposition_")
    assert result["best_candidate"]["family"] == "route_columns_down"
    assert result["finalist_review"][0]["ranking_score"]["primary"]["must_be_supplied_by_agent"] is True
    assert branch.token_order is not None
    assert branch.transform_pipeline["steps"][0]["name"] == "RouteRead"
    assert branch.metadata["cipher_mode"] == "transposition"
    assert branch.metadata["decoded_text"] == plaintext
    assert branch.metadata["pure_transposition_finalist"]["search_session_id"] == result["search_session_id"]
    assert branch.metadata["search_metadata"]["source_branch"] == "main"
    shown = executor._tool_decode_show({"branch": installed["branch"], "count": 1})
    assert shown["rows"][0]["decoded"].startswith("THEQUICKBROWNFOX")

    review = executor._tool_search_review_pure_transposition_finalists({
        "search_session_id": result["search_session_id"],
        "start_rank": 1,
        "count": 2,
    })
    assert review["finalist_review_count"] == 2
    assert review["review_instruction"].startswith("Review these order-only")

    rating = executor._tool_act_rate_transform_finalist({
        "search_session_id": result["search_session_id"],
        "rank": 1,
        "readability_score": 4,
        "label": "coherent_plaintext",
        "rationale": "The preview is fluent English with the expected sentence stream.",
        "coherent_clause": "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    })
    assert rating["session_type"] == "pure_transposition"
    assert rating["finalist"]["ranking_score"]["primary"]["value"] == 4
    assert installed["branch"] in rating["updated_branches"]
    assert branch.metadata["agent_readability_score"] == 4

    installed_again = executor._tool_act_install_pure_transposition_finalists({
        "search_session_id": result["search_session_id"],
        "ranks": [1],
        "branch_prefix": "selected_pure_rank",
    })
    assert installed_again["status"] == "ok"
    selected = executor.workspace.get_branch(installed_again["installed"][0]["branch"])
    assert selected.metadata["pure_transposition_finalist"]["agent_readability_score"] == 4
    assert selected.metadata["decoded_text"] == plaintext


def test_automated_pure_transposition_route_uses_broad_rust_screen(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    k3_path = (
        repo_root
        / "other_tools"
        / "zenith-src"
        / "zenith-inference"
        / "src"
        / "main"
        / "resources"
        / "ciphers"
        / "kryptos3.json"
    )
    payload = json.loads(k3_path.read_text(encoding="utf-8"))
    ciphertext = "".join(ch for ch in payload["ciphertext"] if "A" <= ch <= "Z")
    ct = CipherText(raw=ciphertext, alphabet=Alphabet.from_text(ciphertext), separator=None)
    monkeypatch.setenv("DECIPHER_K3_TRANSMATRIX_MAX_WIDTH", "60")
    monkeypatch.setenv("DECIPHER_PURE_TRANSPOSITION_TOP_N", "5")
    monkeypatch.setenv("DECIPHER_PURE_TRANSPOSITION_THREADS", "2")

    result = run_automated(
        ct,
        language="en",
        cipher_id="kryptos_k3_transmatrix_test",
        cipher_system="pure transposition transmatrix",
    )

    assert result.status == "completed"
    assert result.solver == "pure_transposition_screen_rust"
    assert result.final_decryption.startswith("SLOWLYDESPARATLYSLOWLYTHEREMAINS")
    step = next(step for step in result.steps if step["name"] == "screen_pure_transposition")
    assert step["solver"] == "pure_transposition_screen_rust"
    assert step["selected"]["family"] == "transmatrix"
    assert step["selected"]["pipeline"]["steps"][0]["name"] == "TransMatrix"


def _inverse_pipeline_cipher(plaintext: str, pipeline: TransformPipeline) -> str:
    tokens = [ord(ch) - ord("A") for ch in plaintext]
    order = apply_transform_pipeline(list(range(len(tokens))), pipeline).tokens
    cipher = [0] * len(tokens)
    for output_index, input_index in enumerate(order):
        cipher[input_index] = tokens[output_index]
    return "".join(chr(value + ord("A")) for value in cipher)


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
