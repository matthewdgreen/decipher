"""Composite substitution+transposition peel-and-solve route (Slice C.1, PART 2).

The through-line acceptance fixture is a round-4-class cipher: an English
plaintext put through a monoalphabetic substitution THEN a keyword columnar
transposition (built fresh in-test — the sealed dogfood answer is never read).
It must solve end-to-end through the automated composite route with BOTH layer
keys recovered and recorded.

Firewall: ground truth appears ONLY in the output assertions below. No solving,
detection, routing, peel, or scoring path receives it.
"""
from __future__ import annotations

import inspect
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import automated.runner as runner
from automated.runner import (
    _detect_residual_order,
    _peel_order_layer,
    _run_composite_substitution_transposition,
    _select_solver_path,
    run_automated,
)
from ciphers.transposition import _rank_order, columnar_encrypt
from models.alphabet import Alphabet
from models.cipher_text import CipherText

# ~447 chars of natural English — long enough that the plain substitution anneal
# recovers the full 26-letter key on the (no-boundary) peeled stream.
PLAINTEXT = (
    "WHENINTHECOURSEOFHUMANEVENTSITBECOMESNECESSARYFORONEPEOPLETODISSOLVETHE"
    "POLITICALBANDSWHICHHAVECONNECTEDTHEMWITHANOTHERANDTOASSUMEAMONGTHEPOWERSOF"
    "THEEARTHTHESEPARATEANDEQUALSTATIONTOWHICHTHELAWSOFNATUREANDOFNATURESGODENTITLE"
    "THEMADECENTRESPECTTOTHEOPINIONSOFMANKINDREQUIRESTHATTHEYSHOULDDECLARETHECAUSES"
    "WHICHIMPELTHEMTOTHESEPARATIONWEHOLDTHESETRUTHSTOBESELFEVIDENTTHATALLMENARE"
    "CREATEDEQUALTHATTHEYAREENDOWEDBYTHEIRCREATORWITHCERTAINUNALIENABLERIGHTS"
)

COMPOSITE_KEYWORD = "MASONRY"  # the round-4 keyword (width 7)
SUBST_SEED = 1234


def _random_monoalphabetic(seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    perm = letters[:]
    rng.shuffle(perm)
    return {p: c for p, c in zip(letters, perm)}


def _substitute(plaintext: str, mapping: dict[str, str]) -> str:
    return "".join(mapping[ch] for ch in plaintext)


def _make_round4_class_cipher(keyword: str = COMPOSITE_KEYWORD, seed: int = SUBST_SEED):
    """Build a fresh substitution-THEN-columnar composite (no sealed answer read)."""
    mapping = _random_monoalphabetic(seed)
    substituted = _substitute(PLAINTEXT, mapping)
    ciphertext = columnar_encrypt(substituted, keyword)
    return ciphertext, substituted, mapping


def _az_cipher_text(raw: str) -> CipherText:
    return CipherText(
        raw=raw, alphabet=Alphabet.standard_english(), source="test", separator=None
    )


def _char_accuracy(a: str, b: str) -> float:
    m = min(len(a), len(b))
    if not a and not b:
        return 1.0
    return sum(1 for i in range(m) if a[i] == b[i]) / max(len(a), len(b))


class TestResidualOrderDetection:
    """Detection must separate composite / plain substitution / homophonic."""

    def test_composite_detected(self):
        ciphertext, _substituted, _mapping = _make_round4_class_cipher()
        residual = _detect_residual_order(_az_cipher_text(ciphertext), "en")
        assert residual["applicable"] is True
        assert residual["is_composite"] is True
        assert residual["structure_ratio"] < residual["structure_ratio_absent_threshold"]

    def test_plain_substitution_not_composite(self):
        # A plain monoalphabetic substitution PRESERVES adjacency structure, so the
        # residual signal must NOT fire (the route returns the substitution as-is).
        mapping = _random_monoalphabetic(SUBST_SEED)
        substituted = _substitute(PLAINTEXT, mapping)
        residual = _detect_residual_order(_az_cipher_text(substituted), "en")
        assert residual["applicable"] is True
        assert residual["is_composite"] is False
        assert residual["structure_ratio"] >= residual["structure_ratio_absent_threshold"]

    def test_homophonic_not_applicable(self):
        # A dense (>26-symbol) homophonic inventory is out of scope for the peel
        # (it targets monoalphabetic substitution): detection is "not applicable".
        rng = random.Random(5)
        tokens = [f"S{rng.randint(1, 40):03d}" for _ in range(200)]
        raw = " ".join(tokens)
        alphabet = Alphabet.from_text(raw, multisym=True)
        assert alphabet.size > 26
        cipher = CipherText(raw=raw, alphabet=alphabet, source="test", separator=None)
        residual = _detect_residual_order(cipher, "en")
        assert residual["applicable"] is False
        assert residual["is_composite"] is False


class TestPeelOrderLayer:
    def test_peel_recovers_columnar_order(self):
        ciphertext, substituted, _mapping = _make_round4_class_cipher()
        peel = _peel_order_layer(ciphertext, "en")
        assert peel is not None
        assert peel["kind"] == "columnar"
        assert peel["column_count"] == len(COMPOSITE_KEYWORD)
        assert peel["column_order"] == list(_rank_order(COMPOSITE_KEYWORD))
        # Peeled stream is the (still-substituted) monoalphabetic cipher.
        assert peel["decoded_stream"] == substituted


class TestEndToEndCompositeRoute:
    """The acceptance through-line: round-4-class -> solved, both layers recorded."""

    def test_composite_route_solves_and_records_both_layers(self):
        ciphertext, _substituted, _mapping = _make_round4_class_cipher()
        cipher = _az_cipher_text(ciphertext)
        result = run_automated(
            cipher_text=cipher,
            language="en",
            ground_truth=PLAINTEXT,  # OUTPUT-only firewall assertion
            cipher_system="substitution_transposition_composite",
        )
        # High char accuracy end-to-end (isolation test proved 100% reachable).
        assert result.char_accuracy >= 0.95

        step = next(
            s for s in result.steps
            if isinstance(s, dict) and s.get("name") == "composite_substitution_transposition"
        )
        assert step["outcome"] == "peeled_and_solved"

        # BOTH layer keys recorded.
        transposition = step["transposition"]
        assert transposition["kind"] == "columnar"
        assert transposition["column_count"] == len(COMPOSITE_KEYWORD)
        assert transposition["column_order"] == list(_rank_order(COMPOSITE_KEYWORD))
        assert _rank_order(transposition["keyword"]) == transposition["column_order"]

        substitution = step["substitution"]
        assert substitution["key"], "substitution key must be recorded"
        # A near-complete key (one mapping per distinct ciphertext letter present).
        assert len(substitution["key"]) >= 20

    def test_direct_route_function_returns_composite_solver(self):
        ciphertext, _substituted, _mapping = _make_round4_class_cipher()
        cipher = _az_cipher_text(ciphertext)
        solver, key, decryption, step = _run_composite_substitution_transposition(
            cipher, "en", cipher_id="test"
        )
        assert solver == "composite_substitution_transposition_peel"
        assert _char_accuracy(decryption, PLAINTEXT) >= 0.95
        assert step["transposition"]["column_order"] == list(_rank_order(COMPOSITE_KEYWORD))

    def test_plain_substitution_returned_as_is(self):
        # Feeding a PLAIN monoalphabetic cipher to the composite route must detect
        # "no residual order" and return the substitution result unchanged.
        mapping = _random_monoalphabetic(SUBST_SEED)
        substituted = _substitute(PLAINTEXT, mapping)
        cipher = _az_cipher_text(substituted)
        solver, key, decryption, step = _run_composite_substitution_transposition(
            cipher, "en", cipher_id="test"
        )
        assert step["outcome"] == "no_residual_order_returned_substitution"
        assert "transposition" not in step
        assert solver.startswith("native_substitution")


class TestRouterExplicitName:
    """Slice C.1 wires the route only by EXPLICIT name; auto-detect is Slice B."""

    def test_composite_name_routes_to_composite(self):
        ciphertext, _s, _m = _make_round4_class_cipher()
        routing = _select_solver_path(
            _az_cipher_text(ciphertext), "en",
            cipher_system="substitution_transposition_composite",
        )
        assert routing["route"] == "composite_substitution_transposition"

    def test_columnar_name_still_pure_transposition(self):
        ciphertext, _s, _m = _make_round4_class_cipher()
        routing = _select_solver_path(
            _az_cipher_text(ciphertext), "en", cipher_system="columnar_transposition",
        )
        assert routing["route"] == "pure_transposition"

    def test_unlabelled_composite_does_not_auto_route_here(self):
        # The content-signal auto-selection that would send an UNLABELLED composite
        # here is Slice B and is intentionally NOT wired in this slice.
        ciphertext, _s, _m = _make_round4_class_cipher()
        routing = _select_solver_path(_az_cipher_text(ciphertext), "en", cipher_system="")
        assert routing["route"] != "composite_substitution_transposition"

    def test_homophonic_name_unchanged(self):
        ciphertext, _s, _m = _make_round4_class_cipher()
        routing = _select_solver_path(
            _az_cipher_text(ciphertext), "en", cipher_system="homophonic",
        )
        assert routing["route"] == "homophonic"


class TestFirewall:
    """No ground_truth reachable from detection, peel, or the route (spec §5)."""

    def test_no_ground_truth_parameter_in_route_functions(self):
        for fn in (
            _run_composite_substitution_transposition,
            _detect_residual_order,
            _peel_order_layer,
            runner._solve_substitution_with_rescue,
        ):
            params = set(inspect.signature(fn).parameters)
            assert "ground_truth" not in params, f"{fn.__name__} exposes ground_truth"

    def test_columnar_search_has_no_ground_truth(self):
        from analysis import columnar_search

        params = set(inspect.signature(columnar_search.search_keyed_columnar).parameters)
        assert "ground_truth" not in params
