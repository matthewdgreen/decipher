"""Tests for the no-LLM transposition-family solver.

Cases are generated with the Slice-A cipher classes
(``src/ciphers/transposition.py``) at known keys, run through
``analysis.transposition_solver.solve_transposition``, and checked for a real
solve (char accuracy >= 0.90). Also covers determinism, the time/candidate
budget (no hang), and the routing contract (route defers to the existing
pure-transposition screen; a substitution never triggers a transposition
sweep).
"""
from __future__ import annotations

import random
import time

import pytest

from analysis.transposition_solver import (
    solve_transposition,
    transposition_suspicion,
)
from benchmark.loader import parse_canonical_transcription
from ciphers.textutil import clean
from ciphers.transposition import (
    AmscoCipher,
    ColumnarCipher,
    MyszkowskiCipher,
    NihilistTranspositionCipher,
    RailfenceCipher,
    RedefenceCipher,
    columnar_encrypt,
)

# Two fixed English passages (~200 letters each) so cases are deterministic.
_TEXT_A = (
    "the quick brown fox jumps over the lazy dog while the sun sets slowly "
    "behind the ancient mountains and the river flows gently toward the "
    "distant sea carrying with it the secrets of a thousand forgotten years"
)
_TEXT_B = (
    "in the middle of the great forest there stood a small wooden house "
    "where an old woman lived alone with her cat and spent her quiet days "
    "reading books and tending a garden full of herbs flowers and vegetables"
)


def _clean(text: str) -> str:
    return clean(text)


def _cipher_text(letters: str):
    """Build a CipherText from an A-Z letter stream (canonical space form)."""

    return parse_canonical_transcription(" ".join(letters))


def _char_accuracy(decrypted: str, plaintext: str) -> float:
    dec = "".join(c for c in decrypted.upper() if c.isalpha())
    pt = "".join(c for c in plaintext.upper() if c.isalpha())
    if not pt:
        return 0.0
    matches = sum(1 for i in range(min(len(dec), len(pt))) if dec[i] == pt[i])
    return matches / max(len(dec), len(pt))


def _solve(letters: str, family_hint: str, **kwargs) -> str:
    result = solve_transposition(
        _cipher_text(letters), language="en", family_hint=family_hint, **kwargs
    )
    assert result["status"] == "completed", result
    return result["plaintext"]


# ---------------------------------------------------------------------------
# Per-family solves (>= 3 cases each, char accuracy >= 0.90)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["SECRET", "CIPHERKEY", "TRANSPOSE", "BLACKHAT"],
)
def test_columnar_recovers(key):
    pt = _clean(_TEXT_A)
    ct = ColumnarCipher().encrypt(pt, key)
    dec = _solve(ct, "columnar_transposition")
    assert _char_accuracy(dec, pt) >= 0.90


@pytest.mark.parametrize("rails", [3, 4, 5, 7])
def test_railfence_recovers(rails):
    pt = _clean(_TEXT_A)
    ct = RailfenceCipher().encrypt(pt, rails)
    dec = _solve(ct, "railfence")
    assert _char_accuracy(dec, pt) >= 0.90


@pytest.mark.parametrize(
    "key",
    [(4, (2, 0, 3, 1)), (5, (3, 1, 4, 0, 2)), (3, (1, 2, 0))],
)
def test_redefence_recovers(key):
    pt = _clean(_TEXT_B)
    ct = RedefenceCipher().encrypt(pt, key)
    dec = _solve(ct, "redefence")
    assert _char_accuracy(dec, pt) >= 0.90


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_myszkowski_recovers(seed):
    rng = random.Random(seed)
    key = MyszkowskiCipher().random_key(rng)
    pt = _clean(_TEXT_A if seed % 2 == 0 else _TEXT_B)
    ct = MyszkowskiCipher().encrypt(pt, key)
    dec = _solve(ct, "myszkowski")
    assert _char_accuracy(dec, pt) >= 0.90, f"key={key}"


@pytest.mark.parametrize(
    "key",
    [("SECRET", 1), ("CIPHER", 2), ("MYSTERY", 1)],
)
def test_amsco_recovers(key):
    pt = _clean(_TEXT_B)
    ct = AmscoCipher().encrypt(pt, key)
    dec = _solve(ct, "amsco")
    assert _char_accuracy(dec, pt) >= 0.90, f"key={key}"


@pytest.mark.parametrize("key", ["MONK", "ROADS", "CIPHERX"])
def test_nihilist_recovers(key):
    cipher = NihilistTranspositionCipher()
    padded = cipher.prepare(_TEXT_A, key)  # X-padded to a multiple of n^2
    ct = cipher.encrypt(_TEXT_A, key)
    dec = _solve(ct, "nihilist_transposition")
    assert _char_accuracy(dec, padded) >= 0.90, f"key={key}"


# ---------------------------------------------------------------------------
# Keyed-columnar F2 escalation (width robustness)
# ---------------------------------------------------------------------------


def test_keyed_columnar_f2_recovers_width11():
    """The SA search misses this width-11 keyword columnar; the F2 escalation solves it.

    Fixed plaintext + keyword so the case is deterministic; the F2 hill-climb is
    seeded via ``solve_transposition``'s ``seed`` (passed to
    ``ColumnarSearchConfig``). ``THUNDERBOLT`` (width 11) on this passage is one of
    the width-11 misses the F2 escalation targets: the SA leaves the incumbent
    below the solved dict-rate threshold, and the keyed-columnar search recovers
    the exact plaintext.
    """

    pt = _clean(_TEXT_B)
    ct = columnar_encrypt(pt, "THUNDERBOLT")  # width 11
    # Generous explicit budget: decouples the result from machine load (under
    # heavy CPU contention the default 60s wall-clock could starve the F2
    # escalation and silently skip it — review finding #2).
    result = solve_transposition(
        _cipher_text(ct), language="en", family_hint="columnar_transposition",
        budget_seconds=240.0,
    )
    assert result["status"] == "completed", result
    assert result["plaintext"] == pt
    assert result.get("keyed_columnar_f2", {}).get("adopted") is True, result.get(
        "keyed_columnar_f2"
    )


def test_keyed_columnar_f2_runs_but_does_not_adopt_when_sa_solves():
    """A width the SA already solves: the F2 escalation now runs unconditionally
    but must not displace the (already correct) incumbent.

    S1 dropped the dict-rate skip, so the escalation fires whenever the
    ``columnar`` strategy ran and budget remains. Adoption is strict ``>`` on
    ``full_score``, so an equal/lower-scoring finalist never displaces a solved
    incumbent: the escalation is PRESENT with ``adopted is False`` and the
    returned plaintext is unchanged.
    """

    pt = _clean(_TEXT_A)
    ct = columnar_encrypt(pt, "SECRET")  # width 6, solved by the SA search
    # Generous explicit budget so the SA phase always completes (a truncated SA
    # under contention could leave the incumbent below a real solve).
    # Modest budget (second-review finding #1): F2 is now wall-clock-capped at
    # ~30s, so a huge budget only inflates the SA phase; 60s comfortably covers
    # the width-6 SA solve plus the capped escalation.
    result = solve_transposition(
        _cipher_text(ct), language="en", family_hint="columnar_transposition",
        budget_seconds=60.0,
    )
    assert result["status"] == "completed", result
    kc = result.get("keyed_columnar_f2")
    assert kc is not None, "F2 escalation should run unconditionally now"
    assert kc.get("ran") is True, kc
    assert kc.get("adopted") is False, kc
    # The incumbent solve is untouched — exact plaintext equality (finding #2).
    assert result["plaintext"] == pt


def test_keyed_columnar_f2_skipped_on_tiny_deadline():
    """A tiny remaining budget skips the F2 escalation; the solve still completes."""

    pt = _clean(_TEXT_A)
    ct = columnar_encrypt(pt, "LIGHTHOUSEX")  # width 11 (the SA misses it)
    result = solve_transposition(
        _cipher_text(ct),
        language="en",
        family_hint="columnar_transposition",
        budget_seconds=1.0,
    )
    assert result["status"] == "completed", result
    # Under ~5s remaining => escalation skipped, so it never adopts.
    kc = result.get("keyed_columnar_f2")
    assert kc is None or kc.get("adopted") is not True, kc


def test_keyed_columnar_f2_runs_and_adopts_despite_high_dict_rate_incumbent(monkeypatch):
    """S1: the escalation fires even when the incumbent's dict-rate looks solved.

    Regression for the width-11 miss: the SA can converge to pseudo-English whose
    greedy-segmentation dict_rate crosses the solved bar, which used to *skip* the
    escalation. Here the SA strategy is patched to return a fixed garbage
    candidate and ``_dict_rate`` is forced to 0.99 (so the incumbent reads as
    "already solved"); the keyed-columnar search is patched to return the true
    decode at a strictly better ``full_score``. The escalation must still run and
    adopt.
    """

    import analysis.columnar_search as columnar_search
    import analysis.transposition_solver as ts

    pt = _clean(_TEXT_A)
    ct = columnar_encrypt(pt, "SECRET")

    # SA strategy returns a fixed wrong-basin candidate with a low incumbent score.
    garbage = "X" * len(pt)
    monkeypatch.setattr(
        ts,
        "search_columnar",
        lambda text, language, *, deadline, seed=0: [
            ts._Candidate(garbage, -10.0, "columnar", {"width": 6, "order": list(range(6))})
        ],
    )
    # Force the incumbent to read as "solved" by dict-rate (old skip condition).
    monkeypatch.setattr(ts, "_dict_rate", lambda text, language: 0.99)

    class _FakeFinalist:
        def __init__(self, decoded: str) -> None:
            self.decoded_stream = decoded
            self.column_count = 6
            self.column_order = (0, 1, 2, 3, 4, 5)
            self.keyword = "SECRET"
            self.method = "test_fake"

    monkeypatch.setattr(
        columnar_search,
        "search_keyed_columnar",
        lambda text, scorer, *, config=None: [_FakeFinalist(pt)],
    )

    result = solve_transposition(
        _cipher_text(ct), language="en", family_hint="columnar_transposition",
        budget_seconds=240.0,
    )
    assert result["status"] == "completed", result
    kc = result.get("keyed_columnar_f2")
    assert kc is not None, "escalation must run despite the high-dict-rate incumbent"
    assert kc.get("ran") is True, kc
    assert kc.get("adopted") is True, kc
    assert result["plaintext"] == pt


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_columnar():
    pt = _clean(_TEXT_A)
    ct = ColumnarCipher().encrypt(pt, "SECRET")
    r1 = solve_transposition(_cipher_text(ct), language="en", family_hint="columnar_transposition")
    r2 = solve_transposition(_cipher_text(ct), language="en", family_hint="columnar_transposition")
    assert r1["plaintext"] == r2["plaintext"]
    assert r1["score"] == r2["score"]
    assert r1["family"] == r2["family"]
    assert r1["params"] == r2["params"]


def test_determinism_myszkowski():
    pt = _clean(_TEXT_A)
    ct = MyszkowskiCipher().encrypt(pt, "ABCAB")
    r1 = solve_transposition(_cipher_text(ct), language="en", family_hint="myszkowski")
    r2 = solve_transposition(_cipher_text(ct), language="en", family_hint="myszkowski")
    assert r1["plaintext"] == r2["plaintext"]
    assert r1["score"] == r2["score"]


# ---------------------------------------------------------------------------
# Budget / no-hang
# ---------------------------------------------------------------------------


def test_budget_bounds_runtime():
    """A tiny wall-clock budget must return quickly, never hang."""

    pt = _clean(_TEXT_A + " " + _TEXT_B)  # long, so the full myszkowski search would be slow
    ct = MyszkowskiCipher().encrypt(pt, "EDCBAEDCBA")
    start = time.monotonic()
    result = solve_transposition(
        _cipher_text(ct), language="en", family_hint="myszkowski", budget_seconds=2.0
    )
    elapsed = time.monotonic() - start
    assert result["status"] == "completed"
    assert result["plaintext"]
    # 2s budget + polish/overhead; must be well under the unbounded ~30s search.
    assert elapsed < 12.0, f"budget not respected: {elapsed:.1f}s"


def test_budget_general_cascade_no_hang():
    """Unhinted solve on a hard input still returns within the budget."""

    pt = _clean(_TEXT_A)
    ct = MyszkowskiCipher().encrypt(pt, "CBADCBAD")
    start = time.monotonic()
    result = solve_transposition(
        _cipher_text(ct), language="en", family_hint="", budget_seconds=3.0
    )
    elapsed = time.monotonic() - start
    assert result["status"] == "completed"
    assert elapsed < 15.0, f"budget not respected: {elapsed:.1f}s"


def test_short_input_not_applicable():
    result = solve_transposition(_cipher_text("ABCDE"), language="en")
    assert result["status"] == "not_applicable"


# ---------------------------------------------------------------------------
# Routing contract
# ---------------------------------------------------------------------------


def test_transposition_suspicion_fires_for_transposition():
    pt = _clean(_TEXT_A)
    ct = ColumnarCipher().encrypt(pt, "SECRET")
    signal = transposition_suspicion(_cipher_text(ct), "en")
    assert signal["suspicious"] is True


def test_substitution_not_suspicious():
    """A monoalphabetic substitution must NOT look like a transposition."""

    rng = random.Random(7)
    perm = list(range(26))
    rng.shuffle(perm)
    pt = _clean(_TEXT_A)
    sub = "".join(chr(65 + perm[ord(c) - 65]) for c in pt)
    signal = transposition_suspicion(_cipher_text(sub), "en")
    assert signal["suspicious"] is False


def test_substitution_does_not_route_to_transposition():
    """The runner must not send a substitution to the transposition solver."""

    from automated.runner import _select_solver_path

    rng = random.Random(11)
    perm = list(range(26))
    rng.shuffle(perm)
    pt = _clean(_TEXT_A)
    sub = "".join(chr(65 + perm[ord(c) - 65]) for c in pt)
    route = _select_solver_path(_cipher_text(sub), "en", cipher_system="")
    assert route["route"] != "transposition"


def test_fr_composite_does_not_route_to_composite_peel():
    """S2: the composite CONTENT auto-route is en-only.

    ``order_layer_suspected`` uses an en-calibrated structure threshold, so it is
    disabled for other languages. A substituted-transposition text that routes to
    the composite peel under ``en`` must fall through to the substitution default
    under ``fr`` (whose models/dictionaries are language-correct).
    """

    from automated.runner import _select_solver_path
    from ciphers.transposition import ColumnarCipher

    rng = random.Random(5)
    perm = list(range(26))
    rng.shuffle(perm)
    # Long enough for the order-layer min-token guard (>=150 letters).
    pt = _clean(_TEXT_A + " " + _TEXT_B)
    sub = "".join(chr(65 + perm[ord(c) - 65]) for c in pt)
    composite = ColumnarCipher().encrypt(sub, "MASONRY")
    ct = _cipher_text(composite)

    en_route = _select_solver_path(ct, "en", cipher_system="")
    assert en_route["route"] == "composite_substitution_transposition", en_route

    fr_route = _select_solver_path(ct, "fr", cipher_system="")
    assert fr_route["route"] != "composite_substitution_transposition", fr_route


def test_route_hint_defers_to_pure_transposition():
    """The permutation solver leaves route ciphers to the existing screen."""

    from ciphers.transposition import RouteCipher

    pt = _clean(_TEXT_A)
    ct = RouteCipher().encrypt(pt, {"cols": 6, "route": "spiral"})
    result = solve_transposition(
        _cipher_text(ct), language="en", family_hint="route_transposition"
    )
    assert result["status"] == "not_applicable"


def test_railfence_routes_to_transposition():
    from automated.runner import _select_solver_path

    pt = _clean(_TEXT_A)
    ct = RailfenceCipher().encrypt(pt, 4)
    route = _select_solver_path(_cipher_text(ct), "en", cipher_system="railfence")
    assert route["route"] == "transposition"
