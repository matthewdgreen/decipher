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

    def test_unlabelled_composite_auto_routes_via_content(self):
        # Slice B FLIP: the content-signal auto-selection now sends an UNLABELLED
        # composite here (order_layer_suspected on an A-Z alphabet). This was
        # asserted NOT to happen in C.1; Slice B wires the content route.
        ciphertext, _s, _m = _make_round4_class_cipher()
        routing = _select_solver_path(_az_cipher_text(ciphertext), "en", cipher_system="")
        assert routing["route"] == "composite_substitution_transposition"
        # Routed on ciphertext-derived content, not a cipher_system name.
        assert "residual order layer suspected" in routing["reason"]

    def test_homophonic_name_unchanged(self):
        ciphertext, _s, _m = _make_round4_class_cipher()
        routing = _select_solver_path(
            _az_cipher_text(ciphertext), "en", cipher_system="homophonic",
        )
        assert routing["route"] == "homophonic"


class TestRouterContentAutoRoute:
    """Slice B (§3.1/§3.2): the CONTENT auto-route + regression guards.

    This router is load-bearing — every existing family must be unchanged. The
    composite auto-route keys ONLY on ciphertext-derived signals
    (``transposition_suspicion``/``order_layer_suspected`` + alphabet size);
    ground truth never enters routing (firewall §5).
    """

    def _spaced_cipher_text(self, raw: str, group: int = 5) -> CipherText:
        # Word boundaries every ``group`` letters -> word_groups > 1.
        spaced = " ".join(raw[i:i + group] for i in range(0, len(raw), group))
        return CipherText(
            raw=spaced,
            alphabet=Alphabet.standard_english(),
            source="test",
            separator=" ",
        )

    def test_unlabelled_composite_routes_composite_via_content(self):
        # The MUST-fix: an unlabelled no-boundary composite auto-routes to the
        # C.1 peel via content instead of being hijacked to homophonic.
        ciphertext, _s, _m = _make_round4_class_cipher()
        routing = _select_solver_path(_az_cipher_text(ciphertext), "en", cipher_system="")
        assert routing["route"] == "composite_substitution_transposition"
        assert routing["solver"] == "composite_substitution_transposition_peel"

    def test_explicit_fractionation_label_suppresses_composite_route(self):
        # P3 (declaration_hardening_spec.md §4): an EXPLICIT fractionation label
        # (bifid/trifid/adfgvx/…) trips order_layer_suspected but must NOT route to
        # the composite peel (it honest-fails on fractionation and confuses the
        # agent — fs7). The same content with no label still routes composite; an
        # unrelated label ("unknown") is not suppressed.
        ciphertext, _s, _m = _make_round4_class_cipher()
        cipher = _az_cipher_text(ciphertext)
        # Precondition: the residual-order signal fires for this content.
        from analysis.transposition_solver import transposition_suspicion
        assert transposition_suspicion(cipher, "en")["order_layer_suspected"] is True
        for label in ("bifid", "trifid", "ADFGVX", "adfgx", "polybius", "fractionation"):
            routing = _select_solver_path(cipher, "en", cipher_system=label)
            assert routing["route"] != "composite_substitution_transposition", label
        # No label -> content route still fires.
        assert _select_solver_path(cipher, "en", cipher_system="")["route"] == \
            "composite_substitution_transposition"
        # Unrelated label -> not suppressed (control).
        assert _select_solver_path(cipher, "en", cipher_system="unknown")["route"] == \
            "composite_substitution_transposition"

    def test_plain_substitution_still_routes_substitution(self):
        # REGRESSION GUARD: a plain monoalphabetic substitution has GOOD n-gram
        # structure (order_layer_suspected=False) and, with word boundaries, must
        # STILL route to the substitution path — the composite content branch and
        # the no-boundary homophonic default are both skipped.
        mapping = _random_monoalphabetic(SUBST_SEED)
        substituted = _substitute(PLAINTEXT, mapping)
        routing = _select_solver_path(
            self._spaced_cipher_text(substituted), "en", cipher_system="",
        )
        assert routing["route"] == "substitution"

    def test_plain_homophonic_still_routes_homophonic(self):
        # REGRESSION GUARD: a genuinely dense (>26-symbol) homophonic inventory
        # must STILL route homophonic — the composite peel targets monoalphabetic
        # substitution, so the ``alphabet_size <= pt_alpha.size`` guard keeps it
        # out of the composite branch.
        rng = random.Random(7)
        tokens = [f"S{rng.randint(1, 45):03d}" for _ in range(220)]
        raw = " ".join(tokens)
        alphabet = Alphabet.from_text(raw, multisym=True)
        assert alphabet.size > 26
        cipher = CipherText(raw=raw, alphabet=alphabet, source="test", separator=None)
        routing = _select_solver_path(cipher, "en", cipher_system="")
        assert routing["route"] == "homophonic"

    def test_unlabelled_pure_transposition_not_composite(self):
        # REGRESSION GUARD (the subtle one): an UNLABELLED pure transposition
        # (letters PRESERVED, NOT substituted) ALSO trips order_layer_suspected
        # (scrambled adjacency + language-like SHAPE). It must NOT hit the
        # composite route: its by-letter monogram cosine is high, so the
        # pure-transposition/keyed-column block claims it FIRST via the
        # ``suspicious`` signal, above the composite content branch.
        pure = columnar_encrypt(PLAINTEXT, COMPOSITE_KEYWORD)
        # Precondition: the residual-order signal fires for BOTH pure transposition
        # and the composite — proving the guard is the by-letter cosine ordering,
        # not the residual signal.
        from analysis.transposition_solver import transposition_suspicion

        susp = transposition_suspicion(_az_cipher_text(pure), "en")
        assert susp["order_layer_suspected"] is True
        assert susp["suspicious"] is True  # by-letter cosine high (unsubstituted)

        routing = _select_solver_path(_az_cipher_text(pure), "en", cipher_system="")
        assert routing["route"] != "composite_substitution_transposition"
        assert routing["route"] == "transposition"

    def test_named_no_boundary_vigenere_routes_periodic(self):
        # F1 secondary (spec §3.1 point 3): the Vigenere periodic half is left to
        # the existing cipher_system NAME dispatch (documented, not over-
        # engineered). A NAMED no-boundary Vigenere routes periodic — the composite
        # content branch does not disturb it. The UNLABELLED no-boundary Vigenere
        # content-route is explicitly out of scope for Slice B.
        ciphertext, _s, _m = _make_round4_class_cipher()
        routing = _select_solver_path(
            _az_cipher_text(ciphertext), "en", cipher_system="vigenere",
        )
        assert routing["route"] == "periodic_polyalphabetic"

    def test_unlabelled_composite_solves_end_to_end_via_content(self):
        # ACCEPTANCE: an unlabelled round-4-class composite now routes via content
        # AND solves end-to-end. ground_truth is OUTPUT-only (firewall §5) — it is
        # NOT passed as cipher_system, so routing is purely content-driven.
        ciphertext, _substituted, _mapping = _make_round4_class_cipher()
        cipher = _az_cipher_text(ciphertext)
        result = run_automated(
            cipher_text=cipher,
            language="en",
            ground_truth=PLAINTEXT,  # OUTPUT-only firewall assertion
            cipher_system="",        # UNLABELLED — routing must come from content
        )
        assert result.char_accuracy >= 0.95

        route_step = next(
            s for s in result.steps
            if isinstance(s, dict) and s.get("name") == "route_automated_solver"
        )
        assert route_step["route"] == "composite_substitution_transposition"

        step = next(
            s for s in result.steps
            if isinstance(s, dict) and s.get("name") == "composite_substitution_transposition"
        )
        assert step["outcome"] == "peeled_and_solved"
        assert step["transposition"]["column_order"] == list(_rank_order(COMPOSITE_KEYWORD))
        assert step["substitution"]["key"], "substitution key must be recorded"


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


# --- Fresh-clone model-resolution regression (dogfood round-4 failure) ---------
# Root cause: on a clone WITHOUT the proprietary Zenith CSV, _homophonic_model
# silently dropped to the weak word-list model instead of the BUNDLED
# models/ngram5_en.bin — so hard composites (round-4) produced gibberish. The fix
# adds the bundled binary as a tier, with a case-folding adapter (the binary's
# alphabet is lowercase; the anneal scores uppercase grams).

def test_fresh_clone_uses_bundled_binary_not_word_list(monkeypatch):
    """With the Zenith CSV absent, _homophonic_model MUST pick the bundled binary
    model, not the weak word-list fallback."""
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_MODEL", "/nonexistent/zenith-model.csv")
    from analysis.homophonic import BinaryBackedNGramModel

    model, note = runner._homophonic_model("en", runner._word_list("en"))
    assert isinstance(model, BinaryBackedNGramModel), f"got {type(model).__name__}: {note}"
    assert "bundled binary" in note


def test_binary_adapter_case_folds_and_scores_english(monkeypatch):
    """The adapter must case-fold: the bundled model is lowercase, the anneal
    feeds uppercase. Without the fold every gram floors (real == garbage)."""
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_MODEL", "/nonexistent/z.csv")
    model, _ = runner._homophonic_model("en", runner._word_list("en"))
    assert model.score("THERE") > model.score("QXZJK")  # real 5-gram beats noise
    # the window scorer reads model.log_probs.get(...) directly — same value
    assert model.log_probs.get("THERE", model.floor) == model.score("THERE")


def test_composite_solves_on_bundled_binary_alone(monkeypatch):
    """End-to-end fresh-clone acceptance: the composite route solves with ONLY the
    bundled binary model (Zenith CSV disabled)."""
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_MODEL", "/nonexistent/zenith-model.csv")
    ciphertext, _substituted, _mapping = _make_round4_class_cipher()
    ct = _az_cipher_text(ciphertext)
    _solver, _key, decoded, _step = _run_composite_substitution_transposition(
        ct, language="en", cipher_id="fresh_clone"
    )
    assert _char_accuracy(decoded, PLAINTEXT) >= 0.99, _char_accuracy(decoded, PLAINTEXT)
