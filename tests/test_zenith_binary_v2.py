"""Tests for the ``zenith_binary_v2`` model format (Phase 3a Part A).

Covers:
1. Round-trip build->load for a tiny v2 fixture (Python loader).
2. Shared-semantics between the Python and Rust loaders on a tiny v2 fixture.
3. v1 regression sanity: v1 files still load with a 26-letter alphabet.
4. Lookup-index property test for a non-26 alphabet / non-5 order.
5. Sidecar ``unknown_log_prob`` override.
6. Builder emits v2 (format + alphabet + order) via ``build_model(alphabet=...)``.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from analysis.zenith_solver import (
    ZenithModel,
    _MAGIC_V2,
    _VERSION_V2,
    load_zenith_binary_model,
)
from analysis.polyalphabetic_fast import FAST_AVAILABLE
from tools.corpus.build_model import (
    build_model,
    write_zenith_binary_model_v2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_v2_fixture(
    path: Path,
    alphabet: str,
    order: int,
    values,
    floor: float = 1e-9,
) -> np.ndarray:
    """Write a v2 fixture with the given (alphabet, order) and log-prob values."""
    arr = np.asarray(values, dtype=np.float32)
    assert arr.shape == (len(alphabet) ** order,)
    write_zenith_binary_model_v2(
        path,
        alphabet=alphabet,
        order=order,
        log_probs=arr,
        unknown_log_prob=math.log(floor),
    )
    return arr


# ---------------------------------------------------------------------------
# 1. Round-trip build -> load (Python)
# ---------------------------------------------------------------------------

def test_v2_roundtrip_tiny_fixture(tmp_path):
    alphabet = "xyz"          # 3 symbols
    order = 2                 # 3^2 = 9 entries
    values = [-(i + 1) for i in range(9)]
    path = tmp_path / "tiny_v2.bin"
    arr = _write_v2_fixture(path, alphabet, order, values)

    model = load_zenith_binary_model(path)

    assert model.alphabet == alphabet
    assert model.order == order
    assert model.log_probs.shape == (9,)
    np.testing.assert_allclose(model.log_probs, arr, rtol=0, atol=0)

    # Index math: powers = (3, 1)
    assert math.isclose(model.lookup("xx"), values[0], rel_tol=1e-6)  # 0*3 + 0
    assert math.isclose(model.lookup("xy"), values[1], rel_tol=1e-6)  # 0*3 + 1
    assert math.isclose(model.lookup("xz"), values[2], rel_tol=1e-6)  # 0*3 + 2
    assert math.isclose(model.lookup("yx"), values[3], rel_tol=1e-6)  # 1*3 + 0
    assert math.isclose(model.lookup("zz"), values[8], rel_tol=1e-6)  # 2*3 + 2

    # lookup_indices agrees with lookup
    assert math.isclose(model.lookup_indices([1, 0]), values[3], rel_tol=1e-6)

    # Unknown / out-of-alphabet / wrong-length -> floor
    assert math.isclose(model.unknown_log_prob, math.log(1e-9), rel_tol=1e-5)
    assert model.lookup("xa") == model.unknown_log_prob   # 'a' not in alphabet
    assert model.lookup("x") == model.unknown_log_prob    # wrong length
    assert model.lookup("xxx") == model.unknown_log_prob  # wrong length


def test_v2_header_bytes_match_spec(tmp_path):
    """The written header carries the v2 magic/version/order/alphabet."""
    path = tmp_path / "hdr.bin"
    _write_v2_fixture(path, "abc", 2, list(range(9)))

    with path.open("rb") as fh:
        magic, version, order, alpha_len, alpha_bytes = struct.unpack(">IIIII", fh.read(20))
        alphabet = fh.read(alpha_bytes).decode("utf-8")

    assert magic == _MAGIC_V2
    assert version == _VERSION_V2
    assert order == 2
    assert alpha_len == 3
    assert alphabet == "abc"


# ---------------------------------------------------------------------------
# 2. Shared semantics: Python loader vs Rust loader
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not FAST_AVAILABLE,
    reason="decipher_fast Rust extension is not installed",
)
def test_v2_python_rust_shared_semantics(tmp_path):
    from analysis.polyalphabetic_fast import require_fast

    alphabet = "xyz"
    order = 2
    values = [-1.5, -2.25, -3.0, -0.5, -4.0, -5.5, -6.0, -0.125, -7.0]
    path = tmp_path / "shared_v2.bin"
    _write_v2_fixture(path, alphabet, order, values)

    py_model = load_zenith_binary_model(path)
    fast = require_fast()
    debug = fast.zenith_load_model_debug(str(path))

    assert debug["alphabet"] == py_model.alphabet
    assert debug["order"] == py_model.order
    assert debug["base"] == len(py_model.alphabet)
    assert debug["array_length"] == py_model.log_probs.shape[0]
    assert math.isclose(
        debug["unknown_log_prob"], py_model.unknown_log_prob, rel_tol=1e-6, abs_tol=1e-9
    )
    rust_arr = np.asarray(debug["log_probs"], dtype=np.float64)
    np.testing.assert_allclose(rust_arr, py_model.log_probs.astype(np.float64), rtol=0, atol=1e-7)


@pytest.mark.skipif(
    not FAST_AVAILABLE,
    reason="decipher_fast Rust extension is not installed",
)
def test_v2_python_rust_shared_semantics_non_ascii_order3(tmp_path):
    """Non-ASCII multi-byte alphabet + order 3 still round-trips identically."""
    from analysis.polyalphabetic_fast import require_fast

    alphabet = "äöü"          # each char is 2 UTF-8 bytes
    order = 3                 # 3^3 = 27 entries
    values = [-(i * 0.1 + 0.05) for i in range(27)]
    path = tmp_path / "umlaut_v2.bin"
    _write_v2_fixture(path, alphabet, order, values)

    py_model = load_zenith_binary_model(path)
    assert py_model.alphabet == alphabet
    assert py_model.order == 3
    assert py_model.log_probs.shape == (27,)

    fast = require_fast()
    debug = fast.zenith_load_model_debug(str(path))
    assert debug["alphabet"] == alphabet
    assert debug["order"] == 3
    assert debug["base"] == 3
    rust_arr = np.asarray(debug["log_probs"], dtype=np.float64)
    np.testing.assert_allclose(rust_arr, py_model.log_probs.astype(np.float64), rtol=0, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. v1 regression sanity (loader still handles the classic format)
# ---------------------------------------------------------------------------

def _write_minimal_v1(
    path: Path,
    unknown_prob: float = 1e-9,
    seeds: dict[int, float] | None = None,
) -> None:
    order = 5
    array_len = 26 ** 5
    arr = np.full(array_len, math.log(unknown_prob), dtype=np.float32)
    arr[0] = -1.0
    for idx, val in (seeds or {}).items():
        arr[idx] = val
    with open(path, "wb") as fh:
        fh.write(struct.pack(">IIII", 0x5A4D4D43, 1, order, 3_000_000))
        fh.write(struct.pack(">f", unknown_prob))
        fh.write(struct.pack(">II", array_len, 26))
        for i in range(26):
            fh.write(struct.pack(">H", ord("a") + i))
            fh.write(struct.pack(">q", 1000))
            fh.write(struct.pack(">d", math.log(1.0 / 26)))
        fh.write(struct.pack(">I", array_len))
        fh.write(arr.astype(">f4").tobytes())


def test_v1_still_loads_with_default_alphabet(tmp_path):
    path = tmp_path / "legacy_v1.bin"
    _write_minimal_v1(path)
    model = load_zenith_binary_model(path)
    assert model.alphabet == "abcdefghijklmnopqrstuvwxyz"
    assert model.order == 5
    assert model.log_probs.shape == (26 ** 5,)
    assert math.isclose(model.unknown_log_prob, math.log(1e-9), rel_tol=1e-5)
    # v1 lookup semantics unchanged: "aaaaa" was seeded at index 0.
    assert math.isclose(model.lookup("aaaaa"), -1.0, rel_tol=1e-5)


def test_v1_lookup_index_pin_distinct_positions(tmp_path):
    """Pin lookup("abcde") to its hand-computed flat index.

    All five positions are distinct (a=0, b=1, c=2, d=3, e=4), so a transposed
    power order in the index formula could not accidentally pass:
    idx = 0*26^4 + 1*26^3 + 2*26^2 + 3*26 + 4 = 17576 + 1352 + 78 + 4 = 19010.
    """
    expected_idx = 0 * 26**4 + 1 * 26**3 + 2 * 26**2 + 3 * 26 + 4
    assert expected_idx == 19010
    path = tmp_path / "pin_v1.bin"
    _write_minimal_v1(path, seeds={expected_idx: -2.5})
    model = load_zenith_binary_model(path)
    assert math.isclose(model.lookup("abcde"), -2.5, rel_tol=1e-5)
    assert math.isclose(model.lookup_lo(0, 1, 2, 3, 4), -2.5, rel_tol=1e-5)
    assert math.isclose(model.lookup_indices([0, 1, 2, 3, 4]), -2.5, rel_tol=1e-5)
    # Neighbouring index must NOT alias.
    assert math.isclose(float(model.log_probs[expected_idx]), -2.5, rel_tol=1e-5)
    assert not math.isclose(float(model.log_probs[expected_idx + 1]), -2.5, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 4. Lookup-index property test for a non-26 alphabet / non-5 order
# ---------------------------------------------------------------------------

def test_v2_lookup_index_property_non26(tmp_path):
    """For every gram, lookup returns log_probs at the big-endian flat index."""
    alphabet = "wxyz"         # base 4
    order = 3                 # 4^3 = 64 entries
    base = len(alphabet)
    array_len = base ** order
    # Store the flat index itself as the value so we can assert the mapping.
    values = list(range(array_len))
    path = tmp_path / "prop_v2.bin"
    _write_v2_fixture(path, alphabet, order, values)
    model = load_zenith_binary_model(path)

    powers = [base ** (order - 1 - i) for i in range(order)]
    for i0, c0 in enumerate(alphabet):
        for i1, c1 in enumerate(alphabet):
            for i2, c2 in enumerate(alphabet):
                gram = c0 + c1 + c2
                expected_idx = i0 * powers[0] + i1 * powers[1] + i2 * powers[2]
                assert math.isclose(model.lookup(gram), float(expected_idx), rel_tol=0, abs_tol=0)
                assert math.isclose(
                    model.lookup_indices([i0, i1, i2]), float(expected_idx), abs_tol=0
                )


def test_zenithmodel_default_alphabet_matches_v1_index_math():
    """Constructing a ZenithModel without an alphabet keeps 26^5 index math."""
    log_probs = np.zeros(26 ** 5, dtype=np.float32)
    # Seed "hello" at the classic hardcoded index.
    a, b, c, d, e = (ord(ch) - 97 for ch in "hello")
    idx = a * 456976 + b * 17576 + c * 676 + d * 26 + e
    log_probs[idx] = -3.0
    model = ZenithModel(
        log_probs=log_probs,
        unknown_log_prob=-20.0,
        letter_freq={chr(65 + i): 1.0 / 26 for i in range(26)},
    )
    assert model.alphabet == "abcdefghijklmnopqrstuvwxyz"
    assert math.isclose(model.lookup("hello"), -3.0, rel_tol=1e-5)
    assert math.isclose(model.lookup_lo(a, b, c, d, e), -3.0, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 5. Sidecar unknown_log_prob override
# ---------------------------------------------------------------------------

def test_sidecar_unknown_log_prob_override(tmp_path):
    path = tmp_path / "override_v2.bin"
    _write_v2_fixture(path, "abc", 2, list(range(9)), floor=1e-9)

    # Sidecar carries a different (already-log) floor.
    sidecar = Path(str(path) + ".metadata.json")
    sidecar.write_text(json.dumps({"unknown_log_prob": -12.5}), encoding="utf-8")

    model = load_zenith_binary_model(path)
    assert math.isclose(model.unknown_log_prob, -12.5, rel_tol=1e-9)
    # The floor is what out-of-alphabet lookups return.
    assert math.isclose(model.lookup("a?"), -12.5, rel_tol=1e-9)


def test_no_sidecar_uses_binary_floor(tmp_path):
    path = tmp_path / "nosidecar_v2.bin"
    _write_v2_fixture(path, "abc", 2, list(range(9)), floor=1e-7)
    model = load_zenith_binary_model(path)
    assert math.isclose(model.unknown_log_prob, math.log(1e-7), rel_tol=1e-5)


def test_sidecar_without_unknown_log_prob_keeps_binary_floor(tmp_path):
    path = tmp_path / "partial_sidecar_v2.bin"
    _write_v2_fixture(path, "abc", 2, list(range(9)), floor=1e-9)
    sidecar = Path(str(path) + ".metadata.json")
    sidecar.write_text(json.dumps({"language": "xx"}), encoding="utf-8")
    model = load_zenith_binary_model(path)
    assert math.isclose(model.unknown_log_prob, math.log(1e-9), rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 6. Builder emits v2
# ---------------------------------------------------------------------------

def test_build_model_emits_v2_from_corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    # Only a/b/c survive the alphabet filter; other chars are dropped.
    (corpus_dir / "a.txt").write_text("abcabc ABC! xyz abcab", encoding="utf-8")

    output = tmp_path / "ngram_abc.bin"
    stats = build_model(
        language="en",
        corpus_dir=corpus_dir,
        output_path=output,
        alphabet="abc",
        order=2,
    )

    assert output.exists()
    assert stats.metadata_path.exists()

    metadata = json.loads(stats.metadata_path.read_text(encoding="utf-8"))
    assert metadata["format"] == "zenith_binary_v2"
    assert metadata["alphabet"] == "abc"
    assert metadata["order"] == 2
    assert metadata["array_length"] == 9
    assert metadata["builder_version"] == 2
    assert metadata["normalization"] == {"lowercase": True, "alphabet_filter": True}

    model = load_zenith_binary_model(output)
    assert model.alphabet == "abc"
    assert model.order == 2
    assert model.log_probs.shape == (9,)
    # "ab" occurs (a->b bigram) so it should score above the floor.
    assert model.lookup("ab") > model.unknown_log_prob


def test_build_model_v2_rejects_duplicate_alphabet(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.txt").write_text("aabbcc", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        build_model(
            language="en",
            corpus_dir=corpus_dir,
            output_path=tmp_path / "dup.bin",
            alphabet="aab",
            order=2,
        )


def test_build_model_v2_rejects_non_lowercase_alphabet(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.txt").write_text("abcabc", encoding="utf-8")
    with pytest.raises(ValueError, match="lowercase"):
        build_model(
            language="en",
            corpus_dir=corpus_dir,
            output_path=tmp_path / "upper.bin",
            alphabet="Abc",
            order=2,
        )


# ---------------------------------------------------------------------------
# 7. SA entry-point guards: reject non-v1-shaped models cleanly
# ---------------------------------------------------------------------------

def test_python_zenith_solve_rejects_non_v1_model():
    """A v2-shaped model (base != 26) must raise a clean ValueError, not garbage-score."""
    from analysis.zenith_solver import zenith_solve

    model = ZenithModel(
        log_probs=np.full(9, -5.0, dtype=np.float32),
        unknown_log_prob=-20.0,
        letter_freq={"x": 1 / 3, "y": 1 / 3, "z": 1 / 3},
        order=2,
        alphabet="xyz",
    )
    with pytest.raises(ValueError, match="26-letter order-5"):
        zenith_solve(
            tokens=[0, 1, 2, 0, 1, 2],
            plaintext_ids=[0],
            id_to_letter={0: "A"},
            letter_to_id={"A": 0},
            model=model,
            epochs=1,
            sampler_iterations=5,
            seed=1,
        )


def test_python_zenith_solve_rejects_wrong_order_even_with_26_letters():
    from analysis.zenith_solver import zenith_solve

    model = ZenithModel(
        log_probs=np.full(26 ** 2, -5.0, dtype=np.float32),
        unknown_log_prob=-20.0,
        letter_freq={chr(65 + i): 1.0 / 26 for i in range(26)},
        order=2,
    )
    with pytest.raises(ValueError, match="26-letter order-5"):
        zenith_solve(
            tokens=[0, 1, 2, 0, 1, 2],
            plaintext_ids=[0],
            id_to_letter={0: "A"},
            letter_to_id={"A": 0},
            model=model,
            epochs=1,
            sampler_iterations=5,
            seed=1,
        )


@pytest.mark.skipif(
    not FAST_AVAILABLE,
    reason="decipher_fast Rust extension is not installed",
)
def test_rust_sa_entry_points_reject_v2_model(tmp_path):
    """All three Rust SA entry points must reject a v2 model with a clean error."""
    from analysis.polyalphabetic_fast import require_fast

    fast = require_fast()
    path = tmp_path / "v2_guard.bin"
    _write_v2_fixture(path, "xyz", 2, [-1.0] * 9)

    tokens = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    plaintext_ids = list(range(26))
    id_to_letter = {i: chr(ord("A") + i) for i in plaintext_ids}

    with pytest.raises(ValueError, match="26-letter order-5"):
        fast.zenith_solve_seed(
            str(path), tokens, plaintext_ids, id_to_letter,
            {}, [], 1, 10, 0.012, 0.006, 7, 1,
        )
    with pytest.raises(ValueError, match="26-letter order-5"):
        fast.zenith_transform_candidates_batch(
            str(path), tokens, [], plaintext_ids, id_to_letter,
            1, 10, 0.012, 0.006, [1], 1, 1,
        )
    with pytest.raises(ValueError, match="26-letter order-5"):
        fast.zenith_null_mask_candidates_batch(
            str(path), tokens, [], plaintext_ids, id_to_letter,
            1, 10, 0.012, 0.006, [1], 1, 1,
        )


# ---------------------------------------------------------------------------
# 8. Corrupt v2 headers/payloads are rejected
# ---------------------------------------------------------------------------

def _write_raw_v2(
    path: Path,
    *,
    order: int,
    alphabet: str,
    claimed_array_len: int,
    payload_floats: int,
    alphabet_len: int | None = None,
) -> None:
    """Hand-craft a v2 file so header fields can lie independently of the payload."""
    alphabet_bytes = alphabet.encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack(">I", _MAGIC_V2))
        fh.write(struct.pack(
            ">IIII",
            _VERSION_V2,
            order,
            len(alphabet) if alphabet_len is None else alphabet_len,
            len(alphabet_bytes),
        ))
        fh.write(alphabet_bytes)
        fh.write(struct.pack(">f", 1e-9))
        fh.write(struct.pack(">I", claimed_array_len))
        for _ in range(payload_floats):
            fh.write(struct.pack(">f", -1.0))


def test_v2_rejects_lying_array_length(tmp_path):
    """Header arrayLength disagreeing with base^order must be rejected."""
    path = tmp_path / "lying_len.bin"
    # base=3, order=2 -> expected 9, header claims 10 (payload matches the lie).
    _write_raw_v2(path, order=2, alphabet="abc", claimed_array_len=10, payload_floats=10)
    with pytest.raises(ValueError, match="Unexpected array length"):
        load_zenith_binary_model(path)


def test_v2_rejects_truncated_payload(tmp_path):
    """A payload shorter than the (correct) header arrayLength must be rejected."""
    path = tmp_path / "truncated.bin"
    _write_raw_v2(path, order=2, alphabet="abc", claimed_array_len=9, payload_floats=5)
    with pytest.raises(ValueError, match="Truncated model file"):
        load_zenith_binary_model(path)


def test_v2_rejects_out_of_bounds_order(tmp_path):
    path = tmp_path / "huge_order.bin"
    _write_raw_v2(path, order=13, alphabet="abc", claimed_array_len=9, payload_floats=9)
    with pytest.raises(ValueError, match="Invalid model order"):
        load_zenith_binary_model(path)


def test_v2_rejects_zero_alphabet_len(tmp_path):
    path = tmp_path / "zero_alpha.bin"
    _write_raw_v2(
        path, order=2, alphabet="", claimed_array_len=0, payload_floats=0, alphabet_len=0
    )
    with pytest.raises(ValueError, match="alphabet length"):
        load_zenith_binary_model(path)


@pytest.mark.skipif(
    not FAST_AVAILABLE,
    reason="decipher_fast Rust extension is not installed",
)
def test_v2_rust_rejects_corrupt_headers(tmp_path):
    """The Rust reader mirrors the Python bounds/consistency rejections."""
    from analysis.polyalphabetic_fast import require_fast

    fast = require_fast()

    lying = tmp_path / "lying_len.bin"
    _write_raw_v2(lying, order=2, alphabet="abc", claimed_array_len=10, payload_floats=10)
    with pytest.raises(ValueError, match="Unexpected array length"):
        fast.zenith_load_model_debug(str(lying))

    truncated = tmp_path / "truncated.bin"
    _write_raw_v2(truncated, order=2, alphabet="abc", claimed_array_len=9, payload_floats=5)
    with pytest.raises(ValueError, match="truncated"):
        fast.zenith_load_model_debug(str(truncated))

    huge_order = tmp_path / "huge_order.bin"
    _write_raw_v2(huge_order, order=13, alphabet="abc", claimed_array_len=9, payload_floats=9)
    with pytest.raises(ValueError, match="Invalid model order"):
        fast.zenith_load_model_debug(str(huge_order))


# ---------------------------------------------------------------------------
# 9. CLI option validation (--alphabet / --order / --output)
# ---------------------------------------------------------------------------

def test_cli_alphabet_requires_explicit_output(tmp_path, capsys):
    """--alphabet without --output must error out before building anything."""
    from tools.corpus.__main__ import main

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("abcabc", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["build", "en", "--corpus-dir", str(corpus), "--alphabet", "abc"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "requires an explicit --output" in err


def test_cli_order_requires_alphabet(tmp_path, capsys):
    from tools.corpus.__main__ import main

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("abcabc", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main([
            "build", "en",
            "--corpus-dir", str(corpus),
            "--order", "2",
            "--output", str(tmp_path / "out.bin"),
        ])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--alphabet" in err


def test_cli_alphabet_with_output_still_builds(tmp_path):
    """The happy path (explicit --output) keeps working after the F2 guard."""
    from tools.corpus.__main__ import main

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("abcabc abc abcab", encoding="utf-8")
    output = tmp_path / "ngram_abc.bin"

    rc = main([
        "build", "en",
        "--corpus-dir", str(corpus),
        "--output", str(output),
        "--alphabet", "abc",
        "--order", "2",
    ])
    assert rc == 0
    assert output.exists()
    model = load_zenith_binary_model(output)
    assert model.alphabet == "abc"
    assert model.order == 2
