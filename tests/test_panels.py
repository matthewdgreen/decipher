"""Diagnostic panel tests (INV-0 Part 2 / Part 9)."""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import string

from analysis import panels
from analysis.cipher_id import _chi2_vs_uniform, compute_cipher_fingerprint
from analysis.panels import (
    panel_frequency,
    panel_language,
    panel_numeric_code,
    panel_order_layout,
    panel_periodicity,
    panel_shape,
    run_battery,
)
from ciphers.substitution import SubstitutionCipher
from ciphers.transposition import ColumnarCipher
from models.alphabet import Alphabet


# Firewall: the composite is built in-test by ENCRYPTING a plaintext literal
# (never by reading a sealed answer). Ground truth touches only OUTPUT assertions.
_COMPOSITE_PT = (
    "THEOLDLIGHTHOUSEKEEPERCLIMBEDTHENARROWSTAIRSEACHEVENINGTOTENDTHELAMP"
    "WHOSEBEAMSWEPTACROSSTHEDARKWATERWARNINGSHIPSOFTHEJAGGEDROCKSBELOWAND"
    "HESANGQUIETLYTOHIMSELFASTHEGEARSTURNEDANDTHEGREATLENSREVOLVEDSLOWLY"
    "THROUGHTHECOLDMISTYNIGHTABOVETHESLEEPINGHARBORTOWN"
)


def _substitute(letters, seed):
    """Monoalphabetic substitution via src/ciphers/substitution.py (fixed seed)."""
    alpha = Alphabet(list(string.ascii_uppercase))
    ids = list(range(26))
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    key = dict(zip(ids, shuffled))
    tokens = SubstitutionCipher().encrypt([ord(c) - 65 for c in letters], key, alpha)
    return "".join(chr(65 + t) for t in tokens)


def _order_layout_atoms(rendering):
    res = panel_order_layout([ord(c) - 65 for c in rendering], alphabet_size=26,
                             alphabet_class="letters", language="en",
                             letter_rendering=rendering)
    return res, {a["observation"]: a for a in res.atoms}


def _letters_stream(seed=0, n=300):
    rng = random.Random(seed)
    # skewed letter distribution (English-ish): peaked, small alphabet
    weights = "AAAAEEEEEEETTTTOOONNNIIISSHHRRDLUCMWFGYPBVKJXQZ"
    return [ord(rng.choice(weights)) - 65 for _ in range(n)]


def test_order_layout_not_computable_without_rendering():
    toks = _letters_stream()
    res = panel_order_layout(toks, alphabet_size=26, alphabet_class="symbols",
                             language="en")
    assert res.status == "not_computable"


def test_language_panel_not_computable_for_numeric():
    res = panel_language([1, 2, 3, 4], alphabet_size=26, alphabet_class="numeric",
                         language="en")
    assert res.status == "not_computable"


def test_numeric_panel_not_computable_for_letters():
    res = panel_numeric_code(_letters_stream(), alphabet_size=26,
                             alphabet_class="letters", language="en")
    assert res.status == "not_computable"


def test_periodicity_token_gate():
    res = panel_periodicity([0, 1, 2, 3, 4], alphabet_size=26,
                            alphabet_class="letters", language="en")
    assert res.status == "not_computable"
    assert "too_few" in (res.reason or "")


def test_frequency_adapter_parity_vs_cipher_id():
    toks = _letters_stream(seed=3)
    fp = compute_cipher_fingerprint(toks, max(len(set(toks)), 26), language="en")
    res = panel_frequency(toks, alphabet_size=max(len(set(toks)), 26),
                          alphabet_class="letters", language="en")
    assert abs(res.measurements["ic"] - fp.ic) < 1e-12
    # per-token chi-square flatness == _chi2_vs_uniform(counts, n) / n
    from collections import Counter
    expected_flat = _chi2_vs_uniform(Counter(toks), len(toks)) / len(toks)
    assert abs(res.measurements["flatness"] - expected_flat) < 1e-12


def test_shape_drift_on_spliced_stream():
    # First half all in [0,5], second half all in [20,25] -> high inventory drift.
    rng = random.Random(1)
    spliced = [rng.randint(0, 5) for _ in range(150)] + [rng.randint(20, 25) for _ in range(150)]
    homog = [rng.randint(0, 25) for _ in range(300)]
    d_spliced = panel_shape(spliced, alphabet_size=26, alphabet_class="symbols",
                            language="en").measurements["symbol_inventory_drift_chi2"]
    d_homog = panel_shape(homog, alphabet_size=26, alphabet_class="symbols",
                          language="en").measurements["symbol_inventory_drift_chi2"]
    assert d_spliced > d_homog


def test_battery_exception_isolation(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(panels._PANELS, "frequency", boom)
    battery = run_battery(_letters_stream(), alphabet_size=26,
                          alphabet_class="letters", language="en")
    assert battery["frequency"].status == "not_computable"
    assert "panel_error:RuntimeError" in (battery["frequency"].reason or "")
    # other panels still ran
    assert battery["shape"].status == "ok"


def test_large_symbol_inventory_suppressed_for_numeric():
    # Numeric class must not emit substitution shape atoms.
    res = panel_shape(list(range(60)) * 3, alphabet_size=60, alphabet_class="numeric",
                      language="en")
    assert not any(a["observation"] == "large_symbol_inventory" for a in res.atoms)


# --- composite substitution+transposition residual atom (Slice A §2.1) ---

def test_residual_atom_fires_on_composite_not_on_plain_layers():
    plain = "".join(c for c in _COMPOSITE_PT if "A" <= c <= "Z")
    sub = _substitute(plain, seed=7)
    composite = ColumnarCipher().encrypt(sub, "MASONRY")      # substitute THEN transpose
    transp = ColumnarCipher().encrypt(plain, "MASONRY")       # plain transposition (unsubstituted)

    _, comp_atoms = _order_layout_atoms(composite)
    _, sub_atoms = _order_layout_atoms(sub)
    _, transp_atoms = _order_layout_atoms(transp)

    # Fires on the composite (substituted + order layer)...
    assert "residual_order_after_substitution" in comp_atoms
    atom = comp_atoms["residual_order_after_substitution"]
    assert atom["supports"] == ["substitution_transposition"]
    # ...NOT on a plain substitution (good/preserved n-gram structure)...
    assert "residual_order_after_substitution" not in sub_atoms
    # ...NOT on a plain unsubstituted transposition (language-by-letter monograms).
    assert "residual_order_after_substitution" not in transp_atoms


def test_letters_substituted_drops_transposition_weaken_exactly_when_residual_cofires():
    plain = "".join(c for c in _COMPOSITE_PT if "A" <= c <= "Z")
    sub = _substitute(plain, seed=7)
    composite = ColumnarCipher().encrypt(sub, "MASONRY")

    _, comp_atoms = _order_layout_atoms(composite)
    _, sub_atoms = _order_layout_atoms(sub)

    # Both trip letters_substituted (displaced monograms); only the composite has
    # a residual order layer, so only there does letters_substituted DROP its
    # transposition-weaken.
    assert "letters_substituted" in comp_atoms and "letters_substituted" in sub_atoms
    assert "residual_order_after_substitution" in comp_atoms
    assert "residual_order_after_substitution" not in sub_atoms
    assert comp_atoms["letters_substituted"]["weakens"] == []
    assert sub_atoms["letters_substituted"]["weakens"] == ["transposition"]
