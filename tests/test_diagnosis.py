"""Ranked-diagnosis tests (INV-0 Part 6 / Part 9).

Reuses the RECONCILED source-of-truth fixtures from
``scripts/research/calibrate_inv0_scoring.py`` and runs them through the REAL
``investigation.diagnosis.diagnose`` — so the shipped scorer must reproduce every
tabulated outcome, not just the standalone calibration script.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "research"))

if not os.path.isdir(os.path.join(_REPO, "corpus_data", "en")):
    pytest.skip("corpus_data/en not available", allow_module_level=True)

import random  # noqa: E402
import string  # noqa: E402

import calibrate_inv0_scoring as C  # noqa: E402
from ciphers.substitution import SubstitutionCipher  # noqa: E402
from ciphers.transposition import ColumnarCipher  # noqa: E402
from investigation.diagnosis import diagnose  # noqa: E402
from models.alphabet import Alphabet  # noqa: E402


def _dense(values):
    order = {v: i for i, v in enumerate(sorted(set(values)))}
    return [order[v] for v in values]


def _run(spec):
    ids, rend = spec["gen"]()
    ac = spec["alphabet_class"]
    if ac == "numeric":
        toks = _dense(ids)
        return diagnose(toks, alphabet_size=max(len(set(toks)), 26),
                        alphabet_class="numeric", language="en", numeric_values=ids)
    lr = rend if (ac == "letters" and not spec.get("withhold_rendering")) else None
    uniq = len(set(ids))
    return diagnose(ids, alphabet_size=max(uniq, 26), alphabet_class=ac,
                    language="en", letter_rendering=lr)


FIXTURES = C.build_fixtures()

# atom that must appear in the winner's evidence ("right reason").
_RIGHT_REASON = {
    "A_mono": "peaked_monogram_shape",
    "B_periodic": "periodic_ic_recovery",
    "C_homophonic": "large_symbol_inventory",
    "D_transposition": "letters_unsubstituted",
    "E_numeric_book": "numeric_token_stream",
}


@pytest.mark.parametrize("name", ["A_mono", "B_periodic", "C_homophonic",
                                  "D_transposition", "E_numeric_book"])
def test_confident_fixtures(name):
    spec = FIXTURES[name]
    report = _run(spec)
    top = report.ranked[0]
    assert top.family == spec["expected_top1"]
    assert report.verdict == "confident"
    assert top.confidence == "strong"
    assert _RIGHT_REASON[name] in top.evidence


@pytest.mark.parametrize("name,trigger,rec", [
    ("i_short_mono", "a:token_count<60", "disc_mono_homophonic"),
    ("ii_light_homophonic", "b:margin<0.15", "disc_sub_periodic"),
    ("iii_transp_norender", "d:confusable_discriminator_not_run", "disc_mono_transp"),
    ("iv_uniform_random", "b:margin<0.15", "disc_numeric_book_hoax"),
])
def test_near_miss_fixtures(name, trigger, rec):
    spec = FIXTURES[name]
    report = _run(spec)
    assert report.ranked[0].family == spec["expected_top1"]
    assert report.verdict == "uncertain"
    assert trigger in report.verdict_reasons
    assert report.recommended_next
    assert report.recommended_next[0]["discriminator_id"] == rec


def test_hierarchy_subtype_under_parent_not_in_ranked():
    report = _run(FIXTURES["E_numeric_book"])
    ranked_ids = [f.family for f in report.ranked]
    assert "numeric_word_position" not in ranked_ids
    nb = next(f for f in report.ranked if f.family == "numeric_book_cipher")
    sub = {s.family: s.score for s in nb.subtypes}
    assert sub.get("numeric_word_position") == pytest.approx(0.40, abs=1e-6)


def test_numeric_substitution_rows_carry_counterevidence():
    report = _run(FIXTURES["E_numeric_book"])
    for fd in report.ranked:
        if fd.family in C.SUBSTITUTION_PRIMARIES:
            assert "numeric_inconsistent_with_substitution" in fd.counterevidence
            assert "numeric_token_stream" in fd.counterevidence


def test_every_primary_present_in_ranked():
    report = _run(FIXTURES["A_mono"])
    from investigation.families import PRIMARY_IDS
    assert {f.family for f in report.ranked} == set(PRIMARY_IDS)


def test_beale_acceptance_numeric_over_substitution():
    beale = os.path.expanduser(
        "~/Dropbox/src2/cipher_benchmark/benchmark/unsolved/"
        "sources/famous_short/transcriptions")
    if not os.path.isdir(beale):
        pytest.skip("beale transcriptions not available")
    for name, nb_expected in (("beale_1", 0.85), ("beale_3", 0.50)):
        vals = [int(x) for x in open(f"{beale}/{name}.canonical.txt").read().split()]
        toks = _dense(vals)
        report = diagnose(toks, alphabet_size=max(len(set(toks)), 26),
                          alphabet_class="numeric", language="en", numeric_values=vals)
        score = {f.family: f.score for f in report.ranked}
        assert score["numeric_book_cipher"] == pytest.approx(nb_expected, abs=1e-6)
        for fam in C.SUBSTITUTION_PRIMARIES:
            assert score[fam] == pytest.approx(0.0, abs=1e-6)
        assert report.battery_coverage["numeric_code"] == "ran"


def test_determinism_repeat_run():
    a = _run(FIXTURES["A_mono"]).to_dict()
    b = _run(FIXTURES["A_mono"]).to_dict()
    assert a["ranked"] == b["ranked"]
    assert a["view_hash"] == b["view_hash"]


# --- composite substitution+transposition anti-anchoring (Slice A §2.2 / §6) ---

# Firewall: the round-4-style composite is CONSTRUCTED here by encrypting a
# plaintext literal (a monoalphabetic substitution THEN a columnar transposition,
# fixed seed, no word boundaries). The sealed answer file is never read; ground
# truth appears only in the OUTPUT assertions below.
_ROUND4_STYLE_PT = (
    "THEOLDLIGHTHOUSEKEEPERCLIMBEDTHENARROWSTAIRSEACHEVENINGTOTENDTHELAMP"
    "WHOSEBEAMSWEPTACROSSTHEDARKWATERWARNINGSHIPSOFTHEJAGGEDROCKSBELOWAND"
    "HESANGQUIETLYTOHIMSELFASTHEGEARSTURNEDANDTHEGREATLENSREVOLVEDSLOWLY"
    "THROUGHTHECOLDMISTYNIGHTABOVETHESLEEPINGHARBORTOWN"
)


def _build_composite(seed=7, key="MASONRY"):
    plain = "".join(c for c in _ROUND4_STYLE_PT if "A" <= c <= "Z")
    alpha = Alphabet(list(string.ascii_uppercase))
    ids = list(range(26))
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    sub_tokens = SubstitutionCipher().encrypt(
        [ord(c) - 65 for c in plain], dict(zip(ids, shuffled)), alpha)
    sub = "".join(chr(65 + t) for t in sub_tokens)
    return ColumnarCipher().encrypt(sub, key)   # substitute THEN transpose


def test_composite_not_confidently_misdiagnosed_as_plain_mono():
    composite = _build_composite()
    report = diagnose([ord(c) - 65 for c in composite], alphabet_size=26,
                      alphabet_class="letters", language="en",
                      letter_rendering=composite)
    top2 = [fd.family for fd in report.ranked[:2]]
    # The composite must surface as a ranked top-2 family (the exact failure the
    # INV sweep documented: it was silently lost under a confident plain-mono).
    assert "substitution_transposition" in top2, top2
    # ...and the verdict must NOT be a confident plain-mono anchoring.
    assert not (report.verdict == "confident"
                and report.ranked[0].family == "monoalphabetic_substitution"), (
        report.verdict, report.ranked[0].family)
