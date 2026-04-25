from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from analysis.homophonic import (
    _propose_homophonic_move,
    _targeted_symbol_ids,
    build_continuous_ngram_model,
    homophonic_simulated_anneal,
    load_zenith_csv_model,
    substitution_simulated_anneal,
)
from agent.tools_v2 import WorkspaceToolExecutor
import agent.tools_v2 as tools_v2
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


def test_homophonic_simulated_anneal_solves_tiny_seeded_cipher():
    plaintext = "ABRACADABRAABRACADABRA"
    key = {
        "A": ["01", "02", "03"],
        "B": ["04"],
        "R": ["05"],
        "C": ["06"],
        "D": ["07"],
    }
    counters = {letter: 0 for letter in key}
    cipher_tokens: list[str] = []
    for letter in plaintext:
        choices = key[letter]
        cipher_tokens.append(choices[counters[letter] % len(choices)])
        counters[letter] += 1

    symbols = sorted({token for values in key.values() for token in values})
    alphabet = Alphabet(symbols)
    token_ids = [alphabet.id_for(token) for token in cipher_tokens]
    pt_alpha = Alphabet(list("ABCDR"))
    model = build_continuous_ngram_model([plaintext] * 20, order=4)

    result = homophonic_simulated_anneal(
        tokens=token_ids,
        plaintext_ids=list(range(pt_alpha.size)),
        id_to_letter={i: pt_alpha.symbol_for(i) for i in range(pt_alpha.size)},
        letter_to_id={pt_alpha.symbol_for(i): i for i in range(pt_alpha.size)},
        model=model,
        epochs=6,
        sampler_iterations=300,
        seed=3,
        top_n=3,
    )

    assert result.plaintext == plaintext
    assert result.candidates
    assert result.candidates[0].plaintext == plaintext
    assert len(result.candidates) <= 3
    assert len(result.metadata["epoch_traces"]) == result.epochs
    assert result.metadata["epoch_traces"][0]["epoch"] == 1
    assert "index_of_coincidence" in result.metadata["epoch_traces"][0]
    assert result.metadata["move_telemetry"]["single"]["proposals"] > 0


def test_homophonic_simulated_anneal_supports_zenith_like_scoring_controls():
    plaintext = "ABRACADABRAABRACADABRA"
    key = {
        "A": ["01", "02", "03"],
        "B": ["04"],
        "R": ["05"],
        "C": ["06"],
        "D": ["07"],
    }
    counters = {letter: 0 for letter in key}
    cipher_tokens: list[str] = []
    for letter in plaintext:
        choices = key[letter]
        cipher_tokens.append(choices[counters[letter] % len(choices)])
        counters[letter] += 1

    symbols = sorted({token for values in key.values() for token in values})
    alphabet = Alphabet(symbols)
    token_ids = [alphabet.id_for(token) for token in cipher_tokens]
    pt_alpha = Alphabet(list("ABCDR"))
    model = build_continuous_ngram_model([plaintext] * 20, order=4)

    result = homophonic_simulated_anneal(
        tokens=token_ids,
        plaintext_ids=list(range(pt_alpha.size)),
        id_to_letter={i: pt_alpha.symbol_for(i) for i in range(pt_alpha.size)},
        letter_to_id={pt_alpha.symbol_for(i): i for i in range(pt_alpha.size)},
        model=model,
        epochs=4,
        sampler_iterations=150,
        ioc_weight=1.0,
        score_formula="multiplicative_ioc",
        window_step=2,
        seed=3,
    )

    assert len(result.plaintext) == len(plaintext)
    assert result.metadata["score_formula"] == "multiplicative_ioc"
    assert result.metadata["window_step"] == 2


def test_homophonic_simulated_anneal_supports_exact_zenith_ioc_exponent():
    plaintext = "ABRACADABRAABRACADABRA"
    key = {
        "A": ["01", "02", "03"],
        "B": ["04"],
        "R": ["05"],
        "C": ["06"],
        "D": ["07"],
    }
    counters = {letter: 0 for letter in key}
    cipher_tokens: list[str] = []
    for letter in plaintext:
        choices = key[letter]
        cipher_tokens.append(choices[counters[letter] % len(choices)])
        counters[letter] += 1

    symbols = sorted({token for values in key.values() for token in values})
    alphabet = Alphabet(symbols)
    token_ids = [alphabet.id_for(token) for token in cipher_tokens]
    pt_alpha = Alphabet(list("ABCDR"))
    model = build_continuous_ngram_model([plaintext] * 20, order=4)

    result = homophonic_simulated_anneal(
        tokens=token_ids,
        plaintext_ids=list(range(pt_alpha.size)),
        id_to_letter={i: pt_alpha.symbol_for(i) for i in range(pt_alpha.size)},
        letter_to_id={pt_alpha.symbol_for(i): i for i in range(pt_alpha.size)},
        model=model,
        epochs=4,
        sampler_iterations=150,
        ioc_weight=1.0 / 6.0,
        score_formula="multiplicative_ioc",
        window_step=2,
        seed=3,
    )

    assert len(result.plaintext) == len(plaintext)
    assert result.metadata["score_formula"] == "multiplicative_ioc"
    assert result.metadata["window_step"] == 2


def test_homophonic_simulated_anneal_can_stop_early_via_epoch_callback():
    plaintext = "ABRACADABRAABRACADABRA"
    key = {
        "A": ["01", "02", "03"],
        "B": ["04"],
        "R": ["05"],
        "C": ["06"],
        "D": ["07"],
    }
    counters = {letter: 0 for letter in key}
    cipher_tokens: list[str] = []
    for letter in plaintext:
        choices = key[letter]
        cipher_tokens.append(choices[counters[letter] % len(choices)])
        counters[letter] += 1

    symbols = sorted({token for values in key.values() for token in values})
    alphabet = Alphabet(symbols)
    token_ids = [alphabet.id_for(token) for token in cipher_tokens]
    pt_alpha = Alphabet(list("ABCDR"))
    model = build_continuous_ngram_model([plaintext] * 20, order=4)
    seen_epochs: list[int] = []

    def stop_after_three(epoch_info):
        seen_epochs.append(epoch_info["epoch"])
        return epoch_info["epoch"] >= 3

    result = homophonic_simulated_anneal(
        tokens=token_ids,
        plaintext_ids=list(range(pt_alpha.size)),
        id_to_letter={i: pt_alpha.symbol_for(i) for i in range(pt_alpha.size)},
        letter_to_id={pt_alpha.symbol_for(i): i for i in range(pt_alpha.size)},
        model=model,
        epochs=6,
        sampler_iterations=80,
        seed=3,
        epoch_callback=stop_after_three,
    )

    assert seen_epochs == [1, 2, 3]
    assert result.metadata["stopped_early"] is True
    assert result.metadata["stopped_after_epoch"] == 3


def test_propose_homophonic_move_mixed_profile_can_emit_multi_symbol_moves():
    class StubRandom:
        def __init__(self):
            self._randoms = iter([0.05])
            self._choices = iter([1, "E", "T"])

        def random(self):
            return next(self._randoms)

        def choice(self, seq):
            value = next(self._choices)
            return value

    key = {0: 0, 1: 1}
    id_to_letter = {0: "A", 1: "B", 2: "E", 3: "T"}
    letter_to_id = {v: k for k, v in id_to_letter.items()}

    move_kind, move = _propose_homophonic_move(
        0,
        [0, 1],
        key,
        id_to_letter,
        letter_to_id,
        ["E", "T"],
        StubRandom(),  # type: ignore[arg-type]
        move_profile="mixed_v1",
    )

    assert move_kind == "double_reassign"
    assert move == {0: "E", 1: "T"}


def test_propose_homophonic_move_mixed_v2_can_emit_shared_letter_merge():
    class StubRandom:
        def __init__(self):
            self._randoms = iter([0.15])
            self._choices = iter([1, "E"])

        def random(self):
            return next(self._randoms)

        def choice(self, seq):
            return next(self._choices)

    key = {0: 0, 1: 1}
    id_to_letter = {0: "A", 1: "B", 2: "E"}
    letter_to_id = {v: k for k, v in id_to_letter.items()}

    move_kind, move = _propose_homophonic_move(
        0,
        [0, 1],
        key,
        id_to_letter,
        letter_to_id,
        ["E"],
        StubRandom(),  # type: ignore[arg-type]
        move_profile="mixed_v2",
    )

    assert move_kind == "merge"
    assert move == {0: "E", 1: "E"}


def test_propose_homophonic_move_targeted_falls_back_to_single_for_non_targets():
    class StubRandom:
        def __init__(self):
            self._choices = iter(["E"])

        def random(self):
            return 0.05

        def choice(self, seq):
            return next(self._choices)

    key = {0: 0, 1: 1}
    id_to_letter = {0: "A", 1: "B", 2: "E"}
    letter_to_id = {v: k for k, v in id_to_letter.items()}

    move_kind, move = _propose_homophonic_move(
        0,
        [0, 1],
        key,
        id_to_letter,
        letter_to_id,
        ["E"],
        StubRandom(),  # type: ignore[arg-type]
        move_profile="mixed_v1_targeted",
        targeted_symbol_ids={1},
    )

    assert move_kind == "single"
    assert move == {0: "E"}


def test_targeted_symbol_ids_picks_worst_scoring_symbols():
    occurrences = {0: [0], 1: [1], 2: [2], 3: [3]}
    # Four 2-gram windows with poorer scores on the left.
    window_scores = [-10.0, -9.0, -2.0]

    targeted = _targeted_symbol_ids(
        [0, 1, 2, 3],
        occurrences,
        window_scores,
        text_len=4,
        order=2,
        window_step=1,
    )

    assert 0 in targeted
    assert 1 in targeted


def test_homophonic_simulated_anneal_records_move_profile_metadata():
    plaintext = "ABRACADABRAABRACADABRA"
    key = {
        "A": ["01", "02", "03"],
        "B": ["04"],
        "R": ["05"],
        "C": ["06"],
        "D": ["07"],
    }
    counters = {letter: 0 for letter in key}
    cipher_tokens: list[str] = []
    for letter in plaintext:
        choices = key[letter]
        cipher_tokens.append(choices[counters[letter] % len(choices)])
        counters[letter] += 1

    symbols = sorted({token for values in key.values() for token in values})
    alphabet = Alphabet(symbols)
    token_ids = [alphabet.id_for(token) for token in cipher_tokens]
    pt_alpha = Alphabet(list("ABCDR"))
    model = build_continuous_ngram_model([plaintext] * 20, order=4)

    result = homophonic_simulated_anneal(
        tokens=token_ids,
        plaintext_ids=list(range(pt_alpha.size)),
        id_to_letter={i: pt_alpha.symbol_for(i) for i in range(pt_alpha.size)},
        letter_to_id={pt_alpha.symbol_for(i): i for i in range(pt_alpha.size)},
        model=model,
        epochs=2,
        sampler_iterations=50,
        move_profile="mixed_v1",
        seed=3,
    )

    assert result.metadata["move_profile"] == "mixed_v1"
    assert "move_telemetry" in result.metadata


def test_homophonic_simulated_anneal_records_move_profile_metadata_for_mixed_v2():
    plaintext = "ABRACADABRAABRACADABRA"
    key = {
        "A": ["01", "02", "03"],
        "B": ["04"],
        "R": ["05"],
        "C": ["06"],
        "D": ["07"],
    }
    counters = {letter: 0 for letter in key}
    cipher_tokens: list[str] = []
    for letter in plaintext:
        choices = key[letter]
        cipher_tokens.append(choices[counters[letter] % len(choices)])
        counters[letter] += 1

    symbols = sorted({token for values in key.values() for token in values})
    alphabet = Alphabet(symbols)
    token_ids = [alphabet.id_for(token) for token in cipher_tokens]
    pt_alpha = Alphabet(list("ABCDR"))
    model = build_continuous_ngram_model([plaintext] * 20, order=4)

    result = homophonic_simulated_anneal(
        tokens=token_ids,
        plaintext_ids=list(range(pt_alpha.size)),
        id_to_letter={i: pt_alpha.symbol_for(i) for i in range(pt_alpha.size)},
        letter_to_id={pt_alpha.symbol_for(i): i for i in range(pt_alpha.size)},
        model=model,
        epochs=2,
        sampler_iterations=50,
        move_profile="mixed_v2",
        seed=3,
    )

    assert result.metadata["move_profile"] == "mixed_v2"
    assert "move_telemetry" in result.metadata


def test_homophonic_simulated_anneal_records_move_profile_metadata_for_targeted():
    plaintext = "ABRACADABRAABRACADABRA"
    key = {
        "A": ["01", "02", "03"],
        "B": ["04"],
        "R": ["05"],
        "C": ["06"],
        "D": ["07"],
    }
    counters = {letter: 0 for letter in key}
    cipher_tokens: list[str] = []
    for letter in plaintext:
        choices = key[letter]
        cipher_tokens.append(choices[counters[letter] % len(choices)])
        counters[letter] += 1

    symbols = sorted({token for values in key.values() for token in values})
    alphabet = Alphabet(symbols)
    token_ids = [alphabet.id_for(token) for token in cipher_tokens]
    pt_alpha = Alphabet(list("ABCDR"))
    model = build_continuous_ngram_model([plaintext] * 20, order=4)

    result = homophonic_simulated_anneal(
        tokens=token_ids,
        plaintext_ids=list(range(pt_alpha.size)),
        id_to_letter={i: pt_alpha.symbol_for(i) for i in range(pt_alpha.size)},
        letter_to_id={pt_alpha.symbol_for(i): i for i in range(pt_alpha.size)},
        model=model,
        epochs=2,
        sampler_iterations=50,
        move_profile="mixed_v1_targeted",
        seed=3,
    )

    assert result.metadata["move_profile"] == "mixed_v1_targeted"
    assert result.metadata["target_refresh_interval"] == 500


def test_substitution_simulated_anneal_keeps_bijective_key():
    plaintext = "THEOLDTHEOLDTHEOLD"
    cipher_for_plain = {
        "T": "X",
        "H": "Y",
        "E": "Z",
        "O": "A",
        "L": "B",
        "D": "C",
    }
    ciphertext = "".join(cipher_for_plain[ch] for ch in plaintext)
    alphabet = Alphabet(sorted(set(ciphertext)))
    tokens = [alphabet.id_for(ch) for ch in ciphertext]
    pt_alpha = Alphabet(list("DEHLOT"))
    model = build_continuous_ngram_model([plaintext] * 30, order=3)

    result = substitution_simulated_anneal(
        tokens=tokens,
        plaintext_ids=list(range(pt_alpha.size)),
        id_to_letter={i: pt_alpha.symbol_for(i) for i in range(pt_alpha.size)},
        model=model,
        epochs=4,
        sampler_iterations=300,
        seed=2,
    )

    assert result.plaintext == plaintext
    assert len(set(result.key.values())) == len(result.key)
    assert len(result.metadata["epoch_traces"]) == result.epochs


def test_search_homophonic_anneal_tool_writes_complete_branch(monkeypatch):
    raw = "01 02 03 01 02 03"
    alphabet = Alphabet(["01", "02", "03"])
    ct = CipherText(raw=raw, alphabet=alphabet, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"ABC"},
        word_list=["ABCABCABCABC"],
        pattern_dict={},
    )

    out = ex._tool_search_homophonic_anneal({
        "branch": "main",
        "epochs": 1,
        "sampler_iterations": 20,
        "order": 3,
        "model_path": "word_list",
        "seed": 1,
    })

    assert out["solver"] == "native_homophonic_anneal"
    assert out["model_source"] == "word_list"
    assert ex.workspace.is_complete("main")
    assert len(ex.workspace.get_branch("main").key) == 3
    assert "decoded_preview" in out


def test_search_homophonic_anneal_can_return_candidate_branches():
    raw = "01 02 03 01 02 03"
    alphabet = Alphabet(["01", "02", "03"])
    ct = CipherText(raw=raw, alphabet=alphabet, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"ABC"},
        word_list=["ABCABCABCABC", "ACBACBACB"],
        pattern_dict={},
    )

    out = ex._tool_search_homophonic_anneal({
        "branch": "main",
        "epochs": 3,
        "sampler_iterations": 20,
        "order": 3,
        "model_path": "word_list",
        "seed": 2,
        "top_n": 3,
        "write_candidate_branches": True,
    })

    assert out["candidate_count"] >= 1
    assert out["candidates"][0]["branch"] == "main"
    for candidate in out["candidates"][1:]:
        assert candidate["branch"] in ex.workspace.branch_names()


def test_search_automated_solver_tool_installs_key_from_automated_runner(monkeypatch):
    raw = "01 02 03"
    alphabet = Alphabet(["01", "02", "03"])
    ct = CipherText(raw=raw, alphabet=alphabet, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE"},
        word_list=["THE"],
        pattern_dict={},
    )

    def fake_run_automated(**kwargs):
        result = SimpleNamespace(
            status="completed",
            solver="zenith_native",
            elapsed_seconds=1.25,
            error_message="",
            artifact={
                "key": {"0": 19, "1": 7, "2": 4},
                "steps": [
                    {"name": "route_automated_solver", "route": "homophonic", "solver": "zenith_native"},
                    {"name": "search_homophonic_anneal", "solver": "zenith_native", "model_source": "models/ngram5_en.bin"},
                ],
            },
        )
        return result

    monkeypatch.setattr(tools_v2, "run_automated", fake_run_automated)

    out = ex._tool_search_automated_solver({
        "branch": "main",
        "homophonic_solver": "zenith_native",
    })

    assert out["solver"] == "zenith_native"
    assert out["route_step"]["route"] == "homophonic"
    assert out["primary_step"]["solver"] == "zenith_native"
    assert ex.workspace.is_complete("main")
    assert len(ex.workspace.get_branch("main").key) == 3


def test_search_homophonic_anneal_can_use_zenith_native_profile(monkeypatch):
    raw = "01 02 03 01 02 03"
    alphabet = Alphabet(["01", "02", "03"])
    ct = CipherText(raw=raw, alphabet=alphabet, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"ABC"},
        word_list=["ABCABCABCABC"],
        pattern_dict={},
    )

    monkeypatch.setenv("DECIPHER_NGRAM_MODEL_EN", "models/ngram5_en.bin")

    class FakeModel:
        log_probs = [0.0] * 10

    result = SimpleNamespace(
        plaintext="ABCABC",
        key={0: 0, 1: 1, 2: 2},
        score=-1.23,
        normalized_score=-1.23,
        epochs=2,
        sampler_iterations=50,
        accepted_moves=7,
        improved_moves=3,
        elapsed_seconds=0.4,
        fixed_symbols=0,
        metadata={"cipher_symbols": 3},
        candidates=[SimpleNamespace(epoch=1, score=-1.23, normalized_score=-1.23, plaintext="ABCABC", key={0: 0, 1: 1, 2: 2})],
    )

    monkeypatch.setattr("analysis.zenith_solver.load_zenith_binary_model", lambda path: FakeModel())
    monkeypatch.setattr("analysis.zenith_solver.zenith_solve", lambda **kwargs: result)
    monkeypatch.setattr("automated.runner._zenith_native_model_path", lambda language: Path("models/ngram5_en.bin"))

    out = ex._tool_search_homophonic_anneal({
        "branch": "main",
        "solver_profile": "zenith_native",
        "epochs": 2,
        "sampler_iterations": 50,
        "seed": 1,
    })

    assert out["solver"] == "zenith_native"
    assert out["solver_profile"] == "zenith_native"
    assert out["model_source"].endswith("models/ngram5_en.bin")


def test_search_homophonic_anneal_preserve_existing_passes_fixed_cipher_ids(monkeypatch):
    raw = "01 02 03 01 02 03"
    alphabet = Alphabet(["01", "02", "03"])
    ct = CipherText(raw=raw, alphabet=alphabet, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"ABC"},
        word_list=["ABCABCABCABC"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt_alpha = ws.plaintext_alphabet
    ws.set_mapping("main", alphabet.id_for("01"), pt_alpha.id_for("A"))

    monkeypatch.setenv("DECIPHER_NGRAM_MODEL_EN", "models/ngram5_en.bin")

    class FakeModel:
        log_probs = [0.0] * 10

    seen: dict[str, object] = {}

    result = SimpleNamespace(
        plaintext="ABCABC",
        key={0: 0, 1: 1, 2: 2},
        score=-1.23,
        normalized_score=-1.23,
        epochs=2,
        sampler_iterations=50,
        accepted_moves=7,
        improved_moves=3,
        elapsed_seconds=0.4,
        fixed_symbols=1,
        metadata={"cipher_symbols": 3},
        candidates=[SimpleNamespace(epoch=1, score=-1.23, normalized_score=-1.23, plaintext="ABCABC", key={0: 0, 1: 1, 2: 2})],
    )

    def fake_zenith_solve(**kwargs):
        seen["initial_key"] = dict(kwargs["initial_key"])
        seen["fixed_cipher_ids"] = set(kwargs["fixed_cipher_ids"])
        return result

    monkeypatch.setattr("analysis.zenith_solver.load_zenith_binary_model", lambda path: FakeModel())
    monkeypatch.setattr("analysis.zenith_solver.zenith_solve", fake_zenith_solve)
    monkeypatch.setattr("automated.runner._zenith_native_model_path", lambda language: Path("models/ngram5_en.bin"))
    monkeypatch.setattr("automated.runner._maybe_repair_zenith_native_key", lambda **kwargs: {"applied": False, "reason": "test", "key": dict(kwargs["key"]), "plaintext": kwargs["plaintext"]})
    monkeypatch.setattr("automated.runner._maybe_anchor_refine_zenith_native", lambda **kwargs: {"applied": False, "reason": "test", "key": dict(kwargs["key"]), "plaintext": kwargs["plaintext"], "score": kwargs["anneal_score"]})

    out = ex._tool_search_homophonic_anneal({
        "branch": "main",
        "solver_profile": "zenith_native",
        "preserve_existing": True,
        "epochs": 2,
        "sampler_iterations": 50,
        "seed": 1,
    })

    assert out["preserve_existing"] is True
    assert seen["initial_key"] == {0: pt_alpha.id_for("A")}
    assert seen["fixed_cipher_ids"] == {0}


def test_load_zenith_csv_model_reads_requested_order(tmp_path):
    model_path = tmp_path / "model.csv"
    model_path.write_text(
        '"a","1","10","","","0.1","-2.3"\n'
        '"b","1","5","","","0.1","-2.3"\n'
        '"abc","3","4","0.2","-1.6","0.1","-2.3"\n'
        '"bca","3","2","0.1","-2.1","0.1","-2.3"\n'
        '"abcd","4","1","0.1","-3.0","0.1","-2.3"\n',
        encoding="utf-8",
    )

    model = load_zenith_csv_model(model_path, order=3)

    assert model.order == 3
    assert model.log_probs == {"ABC": -1.6, "BCA": -2.1}
    assert round(model.floor, 6) == round(math.log(1 / 15), 6)
    assert "zenith_csv" in model.source
