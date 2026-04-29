from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.tools_v2 import WorkspaceToolExecutor
from analysis.polyalphabetic import (
    encode_keyed_vigenere_plaintext,
    replay_keyed_vigenere,
    search_keyed_vigenere_alphabet_anneal,
    search_keyed_vigenere,
    search_periodic_polyalphabetic,
)
from automated.runner import AutomatedBenchmarkRunner
from automated.runner import run_automated
from benchmark.loader import parse_canonical_transcription
from frontier.suite import load_frontier_suite, resolve_frontier_case
from testgen.builder import build_test_case
from testgen.cache import PlaintextCache
from testgen.spec import TestSpec as SyntheticTestSpec
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


PLAINTEXT = (
    "THEOLDMANANDTHESEAWASWRITTENBYERNESTHEMINGWAY"
    "THEOLDMANCAUGHTABIGFISHANDRETURNEDHOME"
)
KEY = "LEMON"
KRYPTOS_ALPHABET = "KRYPTOSABCDEFGHIJLMNQUVWXZ"


def _vigenere_encrypt(plaintext: str, key: str) -> str:
    shifts = [ord(ch) - ord("A") for ch in key]
    out = []
    for i, ch in enumerate(plaintext):
        out.append(chr((ord(ch) - ord("A") + shifts[i % len(shifts)]) % 26 + ord("A")))
    return "".join(out)


def _cipher_text() -> CipherText:
    return CipherText(
        raw=_vigenere_encrypt(PLAINTEXT, KEY),
        alphabet=Alphabet.standard_english(),
        separator=None,
    )


def test_periodic_polyalphabetic_search_recovers_vigenere_key_and_text():
    result = search_periodic_polyalphabetic(
        _cipher_text(),
        language="en",
        max_period=8,
        variants=["vigenere"],
        top_n=3,
    )

    best = result["best_candidate"]
    assert result["status"] == "completed"
    assert best["variant"] == "vigenere"
    assert best["period"] == len(KEY)
    assert best["key"] == KEY
    assert best["plaintext"] == PLAINTEXT


def test_keyed_vigenere_replay_matches_kryptos_k1_zenith_vector():
    plaintext = "BETWEENSUBTLESHADINGANDTHEABSENCEOFLIGHTLIESTHENUANCEOFIQLUSION"
    ciphertext = "EMUFPHZLRFAXYUSDJKZLDKRNSHGNFIVJYQTQUXQBQVYUVLLTREVJYQTMKYRDMFD"

    encoded = encode_keyed_vigenere_plaintext(
        plaintext,
        key="PALIMPSEST",
        keyed_alphabet=KRYPTOS_ALPHABET,
    )
    assert encoded == ciphertext

    ct = CipherText(raw=ciphertext, alphabet=Alphabet.standard_english(), separator=None)
    decoded = replay_keyed_vigenere(
        ct,
        key="PALIMPSEST",
        keyed_alphabet=KRYPTOS_ALPHABET,
    )

    assert decoded["status"] == "completed"
    assert decoded["key_type"] == "PeriodicAlphabetKey"
    assert decoded["plaintext"] == plaintext


def test_keyed_vigenere_search_recovers_kryptos_k2_periodic_key_from_candidate_alphabet():
    ciphertext = (
        "VFPJUDEEHZWETZYVGWHKKQETGFQJNCEGGWHKK?DQMCPFQZDQMMIAGPFXHQRLGTIMVMZJANQLVKQEDAGDVFRPJUNGEUNA"
        "QZGZLECGYUXUEENJTBJLBQCRTBJDFHRRYIZETKZEMVDUFKSJHKFWHKUWQLSZFTIHHDDDUVH?DWKBFUFPWNTDFIYCUQZERE"
        "EVLDKFEZMOQQJLTTUGSYQPFEUNLAVIDXFLGGTEZ?FKZBSFDQVGOGIPUFXHHDRKFFHQNTGPUAECNUVPDJMQCLQUMUNEDFQE"
        "LZZVRRGKFFVOEEXBDMVPNFQXEZLGREDNQFMPNZGLFLPMRJQYALMGNUVPDXVKPDQUMEBEDMHDAFMJGZNUPLGEWJLLAETG"
    )
    plaintext = (
        "ITWASTOTALLYINVISIBLEHOWSTHATPOSSIBLETHEYUSEDTHEEARTHSMAGNETICFIELDXTHEINFORMATIONWASGATHERED"
        "ANDTRANSMITTEDUNDERGRUUNDTOANUNKNOWNLOCATIONXDOESLANGLEYKNOWABOUTTHISTHEYSHOULDITSBURIEDOUTTHERE"
        "SOMEWHEREXWHOKNOWSTHEEXACTLOCATIONONLYWWTHISWASHISLASTMESSAGEXTHIRTYEIGHTDEGREESFIFTYSEVENMINUTES"
        "SIXPOINTFIVESECONDSNORTHSEVENTYSEVENDEGREESEIGHTMINUTESFORTYFOURSECONDSWESTIDBYROWS"
    )
    ct = CipherText(raw=ciphertext, alphabet=Alphabet.from_text(ciphertext), separator=None)

    result = search_keyed_vigenere(
        ct,
        language="en",
        periods=list(range(1, 13)),
        keyed_alphabets=[KRYPTOS_ALPHABET],
        top_n=3,
    )

    best = result["best_candidate"]
    assert result["status"] == "completed"
    assert best["variant"] == "keyed_vigenere"
    assert best["key"] == "ABSCISSA"
    assert best["plaintext"] == plaintext


def test_keyed_vigenere_tableau_search_tries_standard_vigenere_first():
    result = search_keyed_vigenere(
        _cipher_text(),
        language="en",
        max_period=8,
        alphabet_keywords=[],
        keyed_alphabets=[],
        include_standard_alphabet=True,
        top_n=3,
    )

    best = result["best_candidate"]
    assert result["status"] == "completed"
    assert result["alphabet_candidates_tested"][0]["candidate_type"] == "standard_alphabet"
    assert result["alphabet_candidates_tested"][0]["keyed_alphabet"] == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert best["key"] == KEY
    assert best["metadata"]["keyed_alphabet"] == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert best["plaintext"] == PLAINTEXT


def test_keyed_vigenere_alphabet_anneal_handles_standard_vigenere_baseline():
    result = search_keyed_vigenere_alphabet_anneal(
        _cipher_text(),
        language="en",
        max_period=8,
        include_standard_alphabet=True,
        steps=0,
        restarts=1,
        top_n=3,
    )

    best = result["best_candidate"]
    assert result["status"] == "completed"
    assert result["solver"] == "keyed_vigenere_alphabet_anneal"
    assert best["key"] == KEY
    assert best["metadata"]["keyed_alphabet"] == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert best["plaintext"] == PLAINTEXT


def test_automated_runner_replays_keyed_kryptos_with_skipped_unknowns():
    ciphertext = (
        "VFPJUDEEHZWETZYVGWHKKQETGFQJNCEGGWHKK?DQMCPFQZDQMMIAGPFXHQRLGTIMVMZJANQLVKQEDAGDVFRPJUNGEUNA"
        "QZGZLECGYUXUEENJTBJLBQCRTBJDFHRRYIZETKZEMVDUFKSJHKFWHKUWQLSZFTIHHDDDUVH?DWKBFUFPWNTDFIYCUQZERE"
        "EVLDKFEZMOQQJLTTUGSYQPFEUNLAVIDXFLGGTEZ?FKZBSFDQVGOGIPUFXHHDRKFFHQNTGPUAECNUVPDJMQCLQUMUNEDFQE"
        "LZZVRRGKFFVOEEXBDMVPNFQXEZLGREDNQFMPNZGLFLPMRJQYALMGNUVPDXVKPDQUMEBEDMHDAFMJGZNUPLGEWJLLAETG"
    )
    plaintext = (
        "ITWASTOTALLYINVISIBLEHOWSTHATPOSSIBLETHEYUSEDTHEEARTHSMAGNETICFIELDXTHEINFORMATIONWASGATHERED"
        "ANDTRANSMITTEDUNDERGRUUNDTOANUNKNOWNLOCATIONXDOESLANGLEYKNOWABOUTTHISTHEYSHOULDITSBURIEDOUTTHERE"
        "SOMEWHEREXWHOKNOWSTHEEXACTLOCATIONONLYWWTHISWASHISLASTMESSAGEXTHIRTYEIGHTDEGREESFIFTYSEVENMINUTES"
        "SIXPOINTFIVESECONDSNORTHSEVENTYSEVENDEGREESEIGHTMINUTESFORTYFOURSECONDSWESTIDBYROWS"
    )
    ct = CipherText(
        raw=ciphertext,
        alphabet=Alphabet.from_text(ciphertext),
        separator=None,
    )

    result = run_automated(
        ct,
        language="en",
        cipher_id="kryptos_k2_keyed_vigenere",
        ground_truth=plaintext,
        cipher_system="kryptos_keyed_vigenere",
        solver_hints={
            "type": "keyed_vigenere",
            "periodic_key": "ABSCISSA",
            "keyed_alphabet": KRYPTOS_ALPHABET,
        },
    )

    assert result.status == "completed"
    assert result.solver == "keyed_vigenere_known_replay"
    assert result.final_decryption == plaintext
    assert result.char_accuracy == 1.0
    step = next(step for step in result.steps if step["name"] == "replay_keyed_vigenere")
    assert step["key_type"] == "PeriodicAlphabetKey"
    assert step["skipped_symbol_count"] == 3


def test_automated_runner_can_force_keyed_kryptos_periodic_key_search(monkeypatch):
    ciphertext = (
        "VFPJUDEEHZWETZYVGWHKKQETGFQJNCEGGWHKK?DQMCPFQZDQMMIAGPFXHQRLGTIMVMZJANQLVKQEDAGDVFRPJUNGEUNA"
        "QZGZLECGYUXUEENJTBJLBQCRTBJDFHRRYIZETKZEMVDUFKSJHKFWHKUWQLSZFTIHHDDDUVH?DWKBFUFPWNTDFIYCUQZERE"
        "EVLDKFEZMOQQJLTTUGSYQPFEUNLAVIDXFLGGTEZ?FKZBSFDQVGOGIPUFXHHDRKFFHQNTGPUAECNUVPDJMQCLQUMUNEDFQE"
        "LZZVRRGKFFVOEEXBDMVPNFQXEZLGREDNQFMPNZGLFLPMRJQYALMGNUVPDXVKPDQUMEBEDMHDAFMJGZNUPLGEWJLLAETG"
    )
    plaintext = (
        "ITWASTOTALLYINVISIBLEHOWSTHATPOSSIBLETHEYUSEDTHEEARTHSMAGNETICFIELDXTHEINFORMATIONWASGATHERED"
        "ANDTRANSMITTEDUNDERGRUUNDTOANUNKNOWNLOCATIONXDOESLANGLEYKNOWABOUTTHISTHEYSHOULDITSBURIEDOUTTHERE"
        "SOMEWHEREXWHOKNOWSTHEEXACTLOCATIONONLYWWTHISWASHISLASTMESSAGEXTHIRTYEIGHTDEGREESFIFTYSEVENMINUTES"
        "SIXPOINTFIVESECONDSNORTHSEVENTYSEVENDEGREESEIGHTMINUTESFORTYFOURSECONDSWESTIDBYROWS"
    )
    ct = CipherText(raw=ciphertext, alphabet=Alphabet.from_text(ciphertext), separator=None)
    monkeypatch.setenv("DECIPHER_KEYED_VIGENERE_MODE", "search")
    monkeypatch.setenv("DECIPHER_POLYALPHABETIC_MAX_PERIOD", "12")

    result = run_automated(
        ct,
        language="en",
        cipher_id="kryptos_k2_keyed_vigenere",
        ground_truth=plaintext,
        cipher_system="kryptos_keyed_vigenere",
        solver_hints={
            "type": "keyed_vigenere",
            "periodic_key": "ABSCISSA",
            "keyed_alphabet": KRYPTOS_ALPHABET,
        },
    )

    assert result.status == "completed"
    assert result.solver == "keyed_vigenere_periodic_key_search"
    assert result.final_decryption == plaintext
    assert result.char_accuracy == 1.0
    step = next(step for step in result.steps if step["name"] == "search_keyed_vigenere")
    assert step["key"] == "ABSCISSA"


def test_automated_runner_tableau_search_recovers_kryptos_k2_without_known_key(monkeypatch):
    ciphertext = (
        "VFPJUDEEHZWETZYVGWHKKQETGFQJNCEGGWHKK?DQMCPFQZDQMMIAGPFXHQRLGTIMVMZJANQLVKQEDAGDVFRPJUNGEUNA"
        "QZGZLECGYUXUEENJTBJLBQCRTBJDFHRRYIZETKZEMVDUFKSJHKFWHKUWQLSZFTIHHDDDUVH?DWKBFUFPWNTDFIYCUQZERE"
        "EVLDKFEZMOQQJLTTUGSYQPFEUNLAVIDXFLGGTEZ?FKZBSFDQVGOGIPUFXHHDRKFFHQNTGPUAECNUVPDJMQCLQUMUNEDFQE"
        "LZZVRRGKFFVOEEXBDMVPNFQXEZLGREDNQFMPNZGLFLPMRJQYALMGNUVPDXVKPDQUMEBEDMHDAFMJGZNUPLGEWJLLAETG"
    )
    plaintext = (
        "ITWASTOTALLYINVISIBLEHOWSTHATPOSSIBLETHEYUSEDTHEEARTHSMAGNETICFIELDXTHEINFORMATIONWASGATHERED"
        "ANDTRANSMITTEDUNDERGRUUNDTOANUNKNOWNLOCATIONXDOESLANGLEYKNOWABOUTTHISTHEYSHOULDITSBURIEDOUTTHERE"
        "SOMEWHEREXWHOKNOWSTHEEXACTLOCATIONONLYWWTHISWASHISLASTMESSAGEXTHIRTYEIGHTDEGREESFIFTYSEVENMINUTES"
        "SIXPOINTFIVESECONDSNORTHSEVENTYSEVENDEGREESEIGHTMINUTESFORTYFOURSECONDSWESTIDBYROWS"
    )
    ct = CipherText(raw=ciphertext, alphabet=Alphabet.from_text(ciphertext), separator=None)
    monkeypatch.setenv("DECIPHER_KEYED_VIGENERE_MODE", "tableau_search")
    monkeypatch.setenv("DECIPHER_KEYED_VIGENERE_TABLEAU_KEYWORDS", "KRYPTOS")
    monkeypatch.setenv("DECIPHER_POLYALPHABETIC_MAX_PERIOD", "12")

    result = run_automated(
        ct,
        language="en",
        cipher_id="kryptos_k2_keyed_vigenere",
        ground_truth=plaintext,
        cipher_system="kryptos_keyed_vigenere",
    )

    assert result.status == "completed"
    assert result.solver == "keyed_vigenere_tableau_search"
    assert result.final_decryption == plaintext
    assert result.char_accuracy == 1.0
    step = next(step for step in result.steps if step["name"] == "search_keyed_vigenere_tableaux")
    assert step["key"] == "ABSCISSA"
    assert step["alphabet_keyword"] == "KRYPTOS"


def test_automated_runner_alphabet_anneal_records_experimental_strategy(monkeypatch):
    monkeypatch.setenv("DECIPHER_KEYED_VIGENERE_MODE", "alphabet_anneal")
    monkeypatch.setenv("DECIPHER_KEYED_VIGENERE_ANNEAL_STEPS", "0")
    monkeypatch.setenv("DECIPHER_KEYED_VIGENERE_ANNEAL_RESTARTS", "1")
    monkeypatch.setenv("DECIPHER_POLYALPHABETIC_MAX_PERIOD", "8")

    result = run_automated(
        _cipher_text(),
        language="en",
        cipher_id="standard_vigenere_as_keyed_tableau",
        ground_truth=PLAINTEXT,
        cipher_system="vigenere",
    )

    assert result.status == "completed"
    assert result.solver == "keyed_vigenere_alphabet_anneal"
    assert result.final_decryption == PLAINTEXT
    step = next(step for step in result.steps if step["name"] == "search_keyed_vigenere_alphabet_anneal")
    assert step["initial_alphabets_tested"][0]["candidate_type"] == "standard_alphabet"


def test_agent_can_observe_cipher_id_and_install_periodic_branch():
    ct = _cipher_text()
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "OLD", "MAN", "AND", "SEA", "WAS", "WRITTEN", "HOME"},
        word_list=["THE", "OLD", "MAN", "AND", "SEA", "WAS", "WRITTEN", "HOME"],
        pattern_dict={},
    )

    fingerprint = ex._tool_observe_cipher_id({"branch": "main", "max_period": 8})
    assert fingerprint["ranked_hypotheses"]
    assert any(row["cipher_mode"] == "polyalphabetic_vigenere" for row in fingerprint["ranked_hypotheses"])

    result = ex._tool_search_periodic_polyalphabetic({
        "branch": "main",
        "max_period": 8,
        "variants": ["vigenere"],
        "top_n": 2,
        "install_top_n": 1,
        "new_branch_prefix": "vig",
    })

    installed = result["installed_branches"][0]
    branch = ex.workspace.get_branch(installed["branch"])
    assert branch.metadata["cipher_mode"] == "periodic_polyalphabetic"
    assert branch.metadata["key_type"] == "PeriodicShiftKey"
    assert branch.metadata["periodic_key"] == KEY
    assert branch.metadata["decoded_text"] == PLAINTEXT

    shown = ex._tool_decode_show({"branch": installed["branch"], "count": 1})
    assert shown["rows"][0]["decoded"].startswith("THEOLDMAN")


def test_agent_observe_cipher_shape_reports_unknown_cipher_structure():
    ex = WorkspaceToolExecutor(
        workspace=Workspace(CipherText(raw="1122334455", alphabet=Alphabet.from_text("1122334455"), separator=None)),
        language="en",
        word_set=set(),
        word_list=[],
        pattern_dict={},
    )

    out = ex._tool_observe_cipher_shape({"branch": "main"})

    assert out["token_count"] == 10
    assert out["even_token_count"] is True
    assert out["single_word_or_no_boundary"] is True
    assert out["coordinate_like_fraction"] == 1.0
    assert "observe_cipher_id" in out["recommended_next_tools"]


def test_agent_can_set_and_adjust_periodic_key_on_branch():
    ct = _cipher_text()
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "OLD", "MAN", "AND", "SEA", "WAS", "WRITTEN", "HOME"},
        word_list=["THE", "OLD", "MAN", "AND", "SEA", "WAS", "WRITTEN", "HOME"],
        pattern_dict={},
    )

    set_key = ex._tool_act_set_periodic_key({
        "branch": "main",
        "key": KEY,
        "variant": "vigenere",
    })

    assert set_key["status"] == "ok"
    assert set_key["key"] == KEY
    assert ex.workspace.get_branch("main").metadata["decoded_text"] == PLAINTEXT

    phases = ex._tool_decode_show_phases({"branch": "main", "sample": 8})
    assert phases["period"] == len(KEY)
    assert phases["phase_rows"][0]["key_symbol"] == "L"
    assert phases["phase_rows"][0]["decoded_sample"]

    changed = ex._tool_act_set_periodic_shift({
        "branch": "main",
        "phase": 0,
        "shift": 0,
    })
    assert changed["status"] == "ok"
    assert changed["key"].startswith("A")
    assert ex.workspace.get_branch("main").metadata["decoded_text"] != PLAINTEXT

    restored = ex._tool_act_adjust_periodic_shift({
        "branch": "main",
        "phase": 0,
        "delta": 11,
    })
    assert restored["key"] == KEY
    assert ex.workspace.get_branch("main").metadata["decoded_text"] == PLAINTEXT


def test_agent_periodic_diagnostic_tools_expose_kasiski_phase_and_shift_hints():
    ct = _cipher_text()
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "OLD", "MAN", "AND", "SEA", "WAS", "WRITTEN", "HOME"},
        word_list=["THE", "OLD", "MAN", "AND", "SEA", "WAS", "WRITTEN", "HOME"],
        pattern_dict={},
    )

    kasiski = ex._tool_observe_kasiski({
        "branch": "main",
        "min_ngram": 3,
        "max_ngram": 4,
        "max_period": 12,
    })
    assert kasiski["status"] == "completed"
    assert kasiski["repeated_sequences"]
    assert "observe_phase_frequency" in kasiski["recommended_next_tools"]

    phase = ex._tool_observe_phase_frequency({
        "branch": "main",
        "period": len(KEY),
        "top_n": 3,
    })
    assert phase["status"] == "completed"
    assert len(phase["phase_rows"]) == len(KEY)
    assert phase["phase_rows"][0]["top_cipher_letters"]

    shifts = ex._tool_observe_periodic_shift_candidates({
        "branch": "main",
        "period": len(KEY),
        "variant": "vigenere",
        "top_n": 3,
    })
    assert shifts["status"] == "completed"
    assert shifts["phase_rows"][0]["top_candidates"][0]["key_symbol"] == "L"
    assert shifts["phase_rows"][1]["top_candidates"][0]["key_symbol"] == "E"


def test_automated_runner_routes_explicit_vigenere_metadata_to_periodic_solver():
    result = run_automated(
        _cipher_text(),
        language="en",
        cipher_id="vigenere_demo",
        ground_truth=PLAINTEXT,
        cipher_system="vigenere",
    )

    assert result.status == "completed"
    assert result.solver == "periodic_polyalphabetic_screen"
    assert result.final_decryption == PLAINTEXT
    assert result.char_accuracy == 1.0
    assert result.artifact["cipher_id_report"]["best_period"] % len(KEY) == 0
    step = next(step for step in result.steps if step["name"] == "search_periodic_polyalphabetic")
    assert step["key"] == KEY


def test_testgen_builds_explicit_periodic_polyalphabetic_case(tmp_path):
    spec = SyntheticTestSpec(
        language="en",
        approx_length=20,
        word_boundaries=False,
        seed=7,
        polyalphabetic_variant="vigenere",
        polyalphabetic_key=KEY,
    )
    cache = PlaintextCache(tmp_path / "cache")
    cache.put(
        spec,
        "THE OLD MAN AND THE SEA WAS WRITTEN BY ERNEST HEMINGWAY "
        "THE OLD MAN CAUGHT A BIG FISH AND RETURNED HOME",
    )

    test_data = build_test_case(spec, cache, api_key="")

    assert test_data.test.test_id == "synth_en_20vignb_s7"
    assert test_data.test.cipher_system == "vigenere"
    assert test_data.plaintext == PLAINTEXT
    assert "|" not in test_data.canonical_transcription

    result = run_automated(
        parse_canonical_transcription(test_data.canonical_transcription),
        language="en",
        cipher_id=test_data.test.test_id,
        ground_truth=test_data.plaintext,
        cipher_system=test_data.test.cipher_system,
    )

    assert result.solver == "periodic_polyalphabetic_screen"
    assert result.char_accuracy == 1.0


def test_polyalphabetic_frontier_ladder_runs_automated_from_cached_plaintext(tmp_path):
    cases = load_frontier_suite("frontier/polyalphabetic_ladder.jsonl")
    cache = PlaintextCache(tmp_path / "cache")
    plaintext = (
        "THE OLD MAN AND THE SEA WAS WRITTEN BY ERNEST HEMINGWAY "
        "THE OLD MAN CAUGHT A BIG FISH AND RETURNED HOME"
    )
    for case in cases:
        assert case.synthetic_spec is not None
        cache.put(case.synthetic_spec, plaintext)

    runner = AutomatedBenchmarkRunner(artifact_dir=tmp_path / "artifacts")
    rows = []
    for case in cases:
        test_data = resolve_frontier_case(
            case,
            benchmark_loader=None,
            cache=cache,
        )
        result = runner.run_test(test_data, language=case.synthetic_spec.language)
        rows.append((case.test.test_id, result.solver, result.char_accuracy))

    assert rows == [
        ("synth_en_120vignb_s21", "periodic_polyalphabetic_screen", 1.0),
        ("synth_en_120bfnb_s22", "periodic_polyalphabetic_screen", 1.0),
        ("synth_en_120vbfnb_s23", "periodic_polyalphabetic_screen", 1.0),
        ("synth_en_120grnb_s24", "periodic_polyalphabetic_screen", 1.0),
    ]
