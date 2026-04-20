from __future__ import annotations

import math

from analysis.homophonic import (
    build_continuous_ngram_model,
    homophonic_simulated_anneal,
    load_zenith_csv_model,
)
from agent.tools_v2 import WorkspaceToolExecutor
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
    )

    assert result.plaintext == plaintext


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
