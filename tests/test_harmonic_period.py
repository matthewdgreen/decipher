"""Harmonic period-folding regression tests (INV-0).

Motivated by the round-6 dogfood trace (investigation bbd8eabb899b): a 566-letter
period-8 Quagmire III whose periodic-IC ladder (8, 16, 24 all elevated) was
mis-summarized as best_period=24 (the sparsest, least reliable multiple).

FIREWALL: the only round-6 input here is the CIPHERTEXT STRING, copied literally
below. No sealed plaintext / key is read or used as an input to any estimator.
See docs/specs/inv0_harmonic_period_spec.md.
"""
from __future__ import annotations

import os
import random
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from analysis import polyalphabetic as P  # noqa: E402
from analysis.cipher_id import (  # noqa: E402
    compute_cipher_fingerprint,
    estimate_fundamental_period,
)
from analysis.panels import panel_periodicity  # noqa: E402
from investigation.diagnosis import diagnose  # noqa: E402
from investigation.families import FAMILY_REGISTRY  # noqa: E402

# --- round-6 ciphertext (566 letters, keyed tableau, true period 8) ---
# CIPHERTEXT ONLY — copied so the test does not depend on the home directory.
ROUND6_CIPHERTEXT = (
    "CYOUPUNPNMCSPUGAQOJCPICASTPJMXNHXMWYDXHVEESZOKETXOVSSJOYJVVIDCDJXKVIPG"
    "DYCORZVXNUIPRQVSBGIZNQDTJFFBKZUQXCJPRIKSCZFBOQMMWCFKSEMJNUJQOZJJNPZRJI"
    "IMBICYOUXUOUJPOXUXOAEORZSENZPERZIXSGEPZAWFWCFXTQVNWAXOJACEUJARLVPYEGPO"
    "RFHWBYOQTZKIZGPJNUCXICJFZZOLBMCYOEKRBNPSOPPUQXFFCZUGDYCYJYNUKNXRNJKKBG"
    "ZORQMWBYXTWSSJYDMVJWPUONBZNSSJVRXOVDIEOJMQOIMMZICYOCXLDUITGJPNWNFVPQVI"
    "WCBOLVPFBGMMWJIJNAEYSDQUKHWBNZCGDAVMLZCXJWBVUEXLDUXMEFHSCWMRTTASVABFYZ"
    "OMEHVYEVPGNRXOOYKSKGENVFZRKMFORQVGDAORHXSENUXMWYSJNAQVSZOGDYCCJGPUNANR"
    "URPUOSIMLSSJOABCTZQJNNCTPFCLBUCYOESJHJDWOYMMBGFQQZCNJDCFKIKWOWBFONVGEG"
    "XCTAVL"
)


def _toks(s: str) -> list[int]:
    return [ord(c) - 65 for c in s.upper() if "A" <= c <= "Z"]


# English-frequency-weighted pseudo-plaintext (deterministic per seed). Only
# used to synthesize ciphertext for tests; never a solving input.
_FREQ = "ETAOINSHRDLUCMWFGYPBVKJXQZ"
_WEIGHTS = [13, 9, 8, 7.5, 7, 6.7, 6.3, 6.1, 6, 4.3, 4, 2.8, 2.8, 2.4, 2.4,
            2, 2, 2, 1.9, 1.5, 1, 0.8, 0.2, 0.15, 0.1, 0.07]


def _pseudo_plaintext(nchars: int, seed: int) -> str:
    r = random.Random(seed)
    return "".join(r.choices(_FREQ, weights=_WEIGHTS, k=nchars))


# ---------------------------------------------------------------------------
# (a) round-6 fixture: must fold to fundamental 8, not the naive argmax 24
# ---------------------------------------------------------------------------

def test_round6_fingerprint_best_period_is_8():
    tokens = _toks(ROUND6_CIPHERTEXT)
    assert len(tokens) == 566
    fp = compute_cipher_fingerprint(tokens, 26, language="en")
    assert fp.best_period == 8, f"expected fundamental 8, got {fp.best_period}"
    # the raw per-period table is preserved and shows the ladder
    assert fp.periodic_ic[8] > fp.ic
    assert fp.periodic_ic[24] > fp.ic  # the sparse multiple is elevated too


def test_round6_estimator_folds_and_reconciles_kasiski():
    tokens = _toks(ROUND6_CIPHERTEXT)
    fp = compute_cipher_fingerprint(tokens, 26, language="en")
    fund, detail = estimate_fundamental_period(
        fp.periodic_ic, len(tokens), kasiski_factors=fp.kasiski_spacing_gcds,
    )
    assert fund == 8
    assert detail["folded"] is True
    assert detail["naive_best_period"] == 24  # the naive argmax we corrected
    assert detail["reason"] == "harmonic_fold"
    assert 16 in detail["harmonic_family"] and 24 in detail["harmonic_family"]
    assert detail["kasiski_corroborates"] is True


def test_round6_panel_reports_fundamental_8():
    tokens = _toks(ROUND6_CIPHERTEXT)
    res = panel_periodicity(tokens, alphabet_size=26, alphabet_class="letters",
                            language="en")
    assert res.measurements["fundamental_period"] == 8
    assert res.measurements["naive_best_period"] == 24
    # raw per-period table preserved
    assert "8" in res.measurements["periodic_ic_table"]


def test_round6_summary_names_fundamental_not_noise():
    tokens = _toks(ROUND6_CIPHERTEXT)
    fp = compute_cipher_fingerprint(tokens, 26, language="en")
    s = fp.natural_language_summary
    assert "period 8" in s.lower()
    # the old bug flagged the sparse peak as noise; the fundamental must not be
    assert "treat this as noise" not in s


# ---------------------------------------------------------------------------
# (b) synthetic Vigenere / Quagmire with known periods
# ---------------------------------------------------------------------------

def _fundamental(ciphertext: str) -> int:
    fp = compute_cipher_fingerprint(_toks(ciphertext), 26, language="en")
    return fp.best_period


def test_plain_vigenere_period_5():
    pt = _pseudo_plaintext(1500, 11)
    shifts = [ord(c) - 65 for c in "LEMON"]
    ct = P.encode_plaintext(pt, shifts, variant="vigenere")
    assert _fundamental(ct) == 5


def test_plain_vigenere_period_6():
    pt = _pseudo_plaintext(1500, 12)
    shifts = [ord(c) - 65 for c in "SECRET"]
    ct = P.encode_plaintext(pt, shifts, variant="vigenere")
    assert _fundamental(ct) == 6


def test_quagmire3_period_8():
    pt = _pseudo_plaintext(1500, 20)
    ct = P.encode_quagmire_plaintext(
        pt, cycleword="WATCHDOG", quagmire_type="quag3", alphabet_keyword="MYSTERY",
    )
    assert _fundamental(ct) == 8


def test_quagmire3_period_4_folds_from_higher_multiple():
    # naive argmax lands on a higher multiple (24); folding must recover 4.
    pt = _pseudo_plaintext(1200, 21)
    ct = P.encode_quagmire_plaintext(
        pt, cycleword="LOOP", quagmire_type="quag3", alphabet_keyword="CIPHER",
    )
    fp = compute_cipher_fingerprint(_toks(ct), 26, language="en")
    fund, detail = estimate_fundamental_period(
        fp.periodic_ic, len(_toks(ct)), kasiski_factors=fp.kasiski_spacing_gcds,
    )
    assert fund == 4
    assert detail["folded"] is True
    assert detail["naive_best_period"] % 4 == 0
    assert detail["naive_best_period"] != 4


# ---------------------------------------------------------------------------
# (c) non-periodicity guard: monoalphabetic must NOT fold to a spurious period
# ---------------------------------------------------------------------------

def test_monoalphabetic_does_not_fold():
    r = random.Random(5)
    perm = list(range(26))
    r.shuffle(perm)
    pt = _pseudo_plaintext(1000, 7)
    ct = "".join(chr(65 + perm[ord(c) - 65]) for c in pt)
    fp = compute_cipher_fingerprint(_toks(ct), 26, language="en")
    fund, detail = estimate_fundamental_period(fp.periodic_ic, len(_toks(ct)))
    assert detail["folded"] is False
    assert detail["reason"] == "no_significant_periodicity"
    # fundamental equals the naive argmax (unchanged behavior)
    assert fund == detail["naive_best_period"]


def test_empty_table_returns_none():
    fund, detail = estimate_fundamental_period({}, 0)
    assert fund is None
    assert detail["folded"] is False


# ---------------------------------------------------------------------------
# (d) quagmire_keyed sequencing signpost
# ---------------------------------------------------------------------------

def test_quagmire_keyed_carries_sequencing_hint():
    spec = FAMILY_REGISTRY["quagmire_keyed"]
    assert spec.sequencing_hint
    lowered = spec.sequencing_hint.lower()
    assert "keyed" in lowered or "quagmire" in lowered
    # every other family keeps the empty default
    others = [f.id for f in FAMILY_REGISTRY.values()
              if f.id != "quagmire_keyed" and f.sequencing_hint]
    assert others == []


def test_diagnosis_surfaces_hint_on_polyalphabetic_subtype():
    # a strongly periodic cipher ranks polyalphabetic_periodic top; the
    # quagmire_keyed subtype under it must surface the sequencing hint.
    pt = _pseudo_plaintext(1500, 20)
    ct = P.encode_quagmire_plaintext(
        pt, cycleword="WATCHDOG", quagmire_type="quag3", alphabet_keyword="MYSTERY",
    )
    report = diagnose(_toks(ct), alphabet_size=26, alphabet_class="letters",
                      language="en", letter_rendering=ct)
    poly = next(f for f in report.ranked if f.family == "polyalphabetic_periodic")
    sub = next(s for s in poly.subtypes if s.family == "quagmire_keyed")
    assert sub.sequencing_hint
    assert "sequencing_hint" in sub.to_dict()
