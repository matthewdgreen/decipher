#!/usr/bin/env python3
"""Executable calibration for INV-0 (investigator mode) diagnosis scoring.

This is the reproducible source of truth for the numbers written into
``docs/specs/investigator_inv0_spec.md`` Parts 5, 6, and 9 and the acceptance
criteria. It implements the Part-6 atom catalog + scoring formula and the
Part-5 island-span rule LITERALLY, imports the REAL ``analysis.cipher_id``
(fingerprint priors, ``_normalized_entropy``, ``_chi2_vs_uniform``), and
evaluates all nine Part-9 fixtures plus acceptance criteria 2 and 5 against
generated cipher streams and the real ``beale_1`` / ``beale_3`` numeric
streams.

Two catalogs are implemented and both are run:

  * ``ORIGINAL`` — the second-review spec exactly as written (normH shape
    thresholds, no numeric-vs-substitution weakening atom, always-emit
    baselined atoms, function-word OR-rescue span rule). Reproduces the
    reviewer's failures.
  * ``FIXED``    — the recalibrated catalog derived here. All nine fixtures +
    acceptance 2/5 pass and the Beale/B2 coherence checks hold.

Run: ``PYTHONPATH=src .venv/bin/python scripts/research/calibrate_inv0_scoring.py``
No network, no LLM, pure local compute.
"""
from __future__ import annotations

import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from analysis.cipher_id import (  # noqa: E402  (real priors + shape math)
    _chi2_vs_uniform,
    _normalized_entropy,
    compute_cipher_fingerprint,
)
from analysis.ic import index_of_coincidence  # noqa: E402
from analysis.ngram import NGRAM_CACHE, normalized_ngram_score  # noqa: E402

LANG_IC_EN = 0.0667
BEALE_DIR = os.path.expanduser(
    "~/Dropbox/src2/cipher_benchmark/benchmark/unsolved/"
    "sources/famous_short/transcriptions"
)
CORPUS_DIR = "corpus_data/en"

# Corpus page ids used to build fixtures / references (all confirmed present).
CORP = [74, 76, 84, 11, 98, 120, 161, 205, 230, 17]


# =====================================================================
# Calibrated constants (FIXED catalog) — the derived numbers
# =====================================================================
# Shape atoms switch from normalized-entropy thresholds (which overlap across
# mono/Vigenere at n~300; see reviewer finding 3) to a per-token chi-square
# flatness statistic  flat = chi2_vs_uniform(counts, n) / n  ~=  k_obs * IC - 1
# (excess collision probability over uniform), which is sample-size robust and
# separates the peaked regime from the rest (post-third-review amendment #4):
#   mono / transposition (natural letter freq)  flat ~ 0.30 - 0.74   (peaked)
#   polyalphabetic period-p mixture             flat ~ 0.12 - 0.36   (neither)
#   homophonic (realistic random assignment)    flat ~ 0.13 - 0.21   (neither)
# NOTE: realistic homophonic and Vigenere OVERLAP on flatness (~0.13-0.21), so
# flat_high_entropy (flat < 0.10) is NOT the homophonic discriminator
# (large_symbol_inventory is) — fixture C correctly does NOT fire flat_high_entropy.
PEAK_FLAT_MIN = 0.28      # flat > this AND unique<=28  -> peaked_monogram_shape
PEAK_UNIQUE_MAX = 28
FLAT_FLAT_MAX = 0.10      # flat < this               -> flat_high_entropy

IC_NEAR = 0.012           # |dIC| < this   -> ic_near_language_reference
IC_DEPRESSED = 0.020      # dIC < -this    -> depressed_ic

# periodicity structural gate (mirrors cipher_id strong-recovery condition)
PERIODIC_RECOVERY_MIN = 0.010
PERIODIC_ABS_MARGIN = 0.015    # best_period_ic >= lang_ic_ref - this
PERIODIC_MIN_COLS = 25

# order_layout thresholds vs an English monogram/quadgram reference
MONO_CHI2_LOW = 300.0     # monogram chi2 vs English below -> letters unchanged
MONO_CHI2_HIGH = 500.0    # above -> letters substituted
QUAD_NLL_SCRAMBLED = -6.0  # quadgram mean-loglik below -> not coherent plaintext

# numeric battery
NUMERIC_UNIQUE_RATE = 0.40     # unique/token > this + numeric -> inconsistency
BOOK_KEYLEN_MAX = 100_000      # max(values) <= this -> plausible key length
FRONTLOAD_MIN = 1.5            # front_loading_index >= this (structural gate)
BENFORD_INCONSISTENT = 0.12    # epsilon_benford >= this -> random-like digit law
CHI2_9DOF_05 = 16.919          # chi-square crit, 9 dof, alpha 0.05
CHI2_10CAT_05 = 16.919         # last digit 0..9 -> 9 dof


# =====================================================================
# Fixture stream generators
# =====================================================================
def load_plain(pgid: int, skip: int = 3000, take: int = 30000) -> list[str]:
    with open(f"{CORPUS_DIR}/pg{pgid}.txt", encoding="utf-8", errors="ignore") as f:
        t = f.read()
    return [c for c in t.upper() if c.isalpha()][skip : skip + take]


def load_prose_words(pgid: int, skip_words: int = 400, n_words: int = 260) -> list[str]:
    """Uppercase word tokens (letters only) from a corpus page, for coherence."""
    with open(f"{CORPUS_DIR}/pg{pgid}.txt", encoding="utf-8", errors="ignore") as f:
        t = f.read()
    words = ["".join(c for c in w if c.isalpha()) for w in t.upper().split()]
    words = [w for w in words if w]
    return words[skip_words : skip_words + n_words]


def gen_mono(letters, seed):
    r = random.Random(seed)
    perm = list(range(26))
    r.shuffle(perm)
    ids = [perm[ord(c) - 65] for c in letters]
    return ids, "".join(chr(65 + i) for i in ids)


def gen_vigenere(letters, klen, seed):
    r = random.Random(seed)
    key = [r.randrange(26) for _ in range(klen)]
    ids = [(ord(c) - 65 + key[i % klen]) % 26 for i, c in enumerate(letters)]
    return ids, "".join(chr(65 + i) for i in ids)


def gen_columnar(letters, ncols, seed):
    r = random.Random(seed)
    order = list(range(ncols))
    r.shuffle(order)
    n = len(letters)
    rows = (n + ncols - 1) // ncols
    pad = list(letters) + ["X"] * (rows * ncols - n)
    grid = [pad[i * ncols : (i + 1) * ncols] for i in range(rows)]
    out = [grid[rr][c] for c in order for rr in range(rows)][:n]
    ids = [ord(c) - 65 for c in out]
    return ids, "".join(out)


def gen_homophonic(letters, nsym, seed):
    # Allocate homophones proportional to letter frequency, then pick uniformly
    # at RANDOM among a letter's homophones each occurrence (real homophonic
    # practice; cyclic assignment would inject a spurious period).
    r = random.Random(seed)
    cnt = Counter(letters)
    present = sorted(cnt)
    alloc = {x: 1 for x in present}
    for _ in range(nsym - len(present)):
        best = max(present, key=lambda x: cnt[x] / alloc[x])
        alloc[best] += 1
    symmap = {}
    nxt = 0
    for x in present:
        symmap[x] = list(range(nxt, nxt + alloc[x]))
        nxt += alloc[x]
    ids = [r.choice(symmap[c]) for c in letters]
    return ids, None


def gen_uniform(n, lo, hi, seed):
    r = random.Random(seed)
    return [r.randint(lo, hi) for _ in range(n)]


def gen_book_word_position(n, booklen, seed, front_bias=0.55):
    """Early-biased word-position book cipher (front-loaded key-text indexing)."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        if r.random() < front_bias:
            out.append(r.randint(1, max(2, booklen // 5)))
        else:
            out.append(r.randint(1, booklen))
    return out


def load_beale(name):
    with open(f"{BEALE_DIR}/{name}.canonical.txt") as f:
        return [int(x) for x in f.read().split()]


# =====================================================================
# English references (built once from corpus)
# =====================================================================
_REF = {}


def english_monogram_prob():
    if "mono" not in _REF:
        letters = []
        for p in [1, 2, 5, 7, 10, 15]:
            letters += load_plain(p, skip=5000, take=40000)
        cnt = Counter(letters)
        tot = sum(cnt.values())
        _REF["mono"] = {chr(65 + i): cnt.get(chr(65 + i), 1) / tot for i in range(26)}
    return _REF["mono"]


def quad_logprobs():
    if "quad" not in _REF:
        _REF["quad"] = NGRAM_CACHE.get("en", 4)
    return _REF["quad"]


def monogram_chi2_vs_english(rendering: str) -> float:
    prob = english_monogram_prob()
    n = len(rendering)
    c = Counter(rendering)
    chi = 0.0
    for i in range(26):
        L = chr(65 + i)
        exp = prob[L] * n
        if exp > 0:
            chi += (c.get(L, 0) - exp) ** 2 / exp
    return chi


# =====================================================================
# Numeric battery (Part 4 subset needed by the atoms)
# =====================================================================
def _first_digit(v):
    return int(str(abs(v))[0])


def _last_digit(v):
    return abs(v) % 10


def benford_epsilon(values):
    n = len(values)
    fd = Counter(_first_digit(v) for v in values)
    obs = {d: fd.get(d, 0) / n for d in range(1, 10)}
    ben = {d: math.log10(1 + 1 / d) for d in range(1, 10)}
    return max(abs(obs[d] - ben[d]) for d in range(1, 10))


def last_digit_chi2(values):
    n = len(values)
    ld = Counter(_last_digit(v) for v in values)
    exp = n / 10.0
    return sum((ld.get(d, 0) - exp) ** 2 / exp for d in range(10))


def front_loading_index(values):
    mx = max(values)
    share = sum(1 for v in values if v <= mx / 5.0) / len(values)
    return share / 0.2


def repeat_rate(values):
    n = len(values)
    c = Counter(values)
    return sum(cnt for cnt in c.values() if cnt >= 2) / n


def iid_uniform_repeat_null(values, seed=0, draws=60):
    """Mean+std repeat-rate under an iid-uniform-over-[min,max] null."""
    n = len(values)
    lo, hi = min(values), max(values)
    rng = random.Random(seed)
    rates = []
    for _ in range(draws):
        d = [rng.randint(lo, hi) for _ in range(n)]
        c = Counter(d)
        rates.append(sum(cnt for cnt in c.values() if cnt >= 2) / n)
    m = sum(rates) / len(rates)
    sd = (sum((x - m) ** 2 for x in rates) / len(rates)) ** 0.5
    return m, sd


def best_period_ic(tokens, max_period=26):
    """Max over periods k>=2 of the mean IC of every-k-th-token streams."""
    n = len(tokens)
    uniq = len(set(tokens))
    best = 0.0
    for k in range(2, min(max_period + 1, n // 4 + 2)):
        streams = [tokens[i::k] for i in range(k)]
        ics = [index_of_coincidence(s, max(uniq, 26)) for s in streams if len(s) >= 6]
        if ics:
            best = max(best, sum(ics) / len(ics))
    return best


def periodic_recovery_significant(tokens, fp, seed=0, n_shuffles=300):
    """FIXED gate for periodic_ic_recovery (finding 6 — emit only when a real
    recovery is confirmed by a frequency-preserving shuffle null)."""
    if fp.best_period_ic is None or math.isnan(fp.ic):
        return False, 1.0
    bpi = fp.best_period_ic
    recovery = bpi - fp.ic
    if not (recovery > PERIODIC_RECOVERY_MIN and bpi >= LANG_IC_EN - PERIODIC_ABS_MARGIN):
        return False, 1.0
    rng = random.Random(seed)
    pool = list(tokens)
    ge = 0
    for _ in range(n_shuffles):
        rng.shuffle(pool)
        if best_period_ic(pool) >= bpi:
            ge += 1
    p = (ge + 1) / (n_shuffles + 1)
    return (p <= 0.05), p


def front_loading_significant(values, index, seed=0, draws=300):
    """FIXED gate for front_loading_present (Part-4 baselined:upper vs an iid
    uniform draw — the shuffle null is order-invariant so a parametric uniform
    null is used, matching the spec's 'excess baselined:upper vs a uniform draw')."""
    if index <= 1.0:
        return False, 1.0
    n = len(values)
    lo, hi = min(values), max(values)
    rng = random.Random(seed)
    ge = 0
    for _ in range(draws):
        d = [rng.randint(lo, hi) for _ in range(n)]
        if front_loading_index(d) >= index:
            ge += 1
    p = (ge + 1) / (draws + 1)
    return (p <= 0.05), p


def longest_monotone_run(values):
    best = cur = 1
    for i in range(1, len(values)):
        if values[i] >= values[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def monotone_run_excess_p(values, seed=0, n_shuffles=200):
    """Baselined upper-tail p for longest monotone run vs frequency-preserving
    shuffle null (Gillogly fabrication signature)."""
    obs = longest_monotone_run(values)
    rng = random.Random(seed)
    ge = 0
    pool = list(values)
    for _ in range(n_shuffles):
        rng.shuffle(pool)
        if longest_monotone_run(pool) >= obs:
            ge += 1
    return (ge + 1) / (n_shuffles + 1)


def numeric_battery(values):
    m, sd = iid_uniform_repeat_null(values)
    rr = repeat_rate(values)
    ld = last_digit_chi2(values)
    return {
        "max": max(values),
        "min": min(values),
        "unique_rate": len(set(values)) / len(values),
        "benford_eps": benford_epsilon(values),
        "last_digit_chi2": ld,
        "last_digit_uniform": ld < CHI2_10CAT_05,          # p > 0.05
        "front_loading_index": front_loading_index(values),
        "repeat_rate": rr,
        "repeat_null_mean": m,
        "repeat_null_std": sd,
        "repeat_within_1sd": (sd > 0 and abs(rr - m) <= sd),
        "monotone_run_excess_p": monotone_run_excess_p(values),
    }


# =====================================================================
# Family registry (fields needed for scoring / verdict)
# =====================================================================
PRIMARIES = [
    "monoalphabetic_substitution",
    "homophonic_substitution",
    "polyalphabetic_periodic",
    "transposition",
    "transposition_homophonic",
    "fractionation_transposition",
    "playfair",
    "polygraphic_substitution",
    "nomenclator_codebook",
    "numeric_book_cipher",
    "plaintext_or_hoax",
    "unknown_custom",
]
SUBTYPES = {
    "quagmire_keyed": "polyalphabetic_periodic",
    "numeric_word_position": "numeric_book_cipher",
    "numeric_char_position": "numeric_book_cipher",
    "numeric_skip_nth_word": "numeric_book_cipher",
}
MODIFIERS = ["nulls_noise_layer"]
SUBSTITUTION_PRIMARIES = [
    "monoalphabetic_substitution",
    "homophonic_substitution",
    "polyalphabetic_periodic",
    "transposition",
    "transposition_homophonic",
    "fractionation_transposition",
    "playfair",
    "polygraphic_substitution",
]

CONFUSABLE = {
    "monoalphabetic_substitution": {
        "transposition",
        "homophonic_substitution",
        "polyalphabetic_periodic",
    },
    "transposition": {"monoalphabetic_substitution"},
    "homophonic_substitution": {
        "monoalphabetic_substitution",
        "transposition_homophonic",
    },
    "polyalphabetic_periodic": {"monoalphabetic_substitution"},
    "transposition_homophonic": {"homophonic_substitution"},
    "numeric_book_cipher": {"plaintext_or_hoax"},
    "plaintext_or_hoax": {"numeric_book_cipher"},
}

# discriminator id -> (splits frozenset, depends_on_panels, status)
DISCRIMINATORS = {
    "disc_mono_transp": (
        frozenset({"monoalphabetic_substitution", "transposition"}),
        ("order_layout",),
        "available",
    ),
    "disc_mono_homophonic": (
        frozenset({"monoalphabetic_substitution", "homophonic_substitution"}),
        ("shape", "frequency"),
        "available",
    ),
    "disc_sub_periodic": (
        frozenset({"monoalphabetic_substitution", "polyalphabetic_periodic"}),
        ("periodicity",),
        "available",
    ),
    "disc_periodic_quagmire": (
        frozenset({"polyalphabetic_periodic", "quagmire_keyed"}),
        ("periodicity",),
        "available",
    ),
    "disc_homo_transphomo": (
        frozenset({"homophonic_substitution", "transposition_homophonic"}),
        ("order_layout", "polygraphic"),
        "available",
    ),
    "disc_numeric_book_hoax": (
        frozenset({"numeric_book_cipher", "plaintext_or_hoax"}),
        ("numeric_code",),
        "available",
    ),
    "disc_book_word_char": (
        frozenset({"numeric_word_position", "numeric_char_position"}),
        ("numeric_code",),
        "available",
    ),
}
# family -> its discriminators
FAMILY_DISCS = {}
for _did, (_splits, _dep, _st) in DISCRIMINATORS.items():
    for _fam in _splits:
        FAMILY_DISCS.setdefault(_fam, []).append(_did)

SUSPICION_TO_FAMILY = {
    "monoalphabetic_substitution": "monoalphabetic_substitution",
    "homophonic_substitution": "homophonic_substitution",
    "polyalphabetic_vigenere": "polyalphabetic_periodic",
    "transposition": "transposition",
    "transposition_homophonic": "transposition_homophonic",
    "playfair": "playfair",
}


# =====================================================================
# Atom builders — ORIGINAL and FIXED
# =====================================================================
def atom(obs, weight, rel, supports, weakens, panel="", note=""):
    return {
        "observation": obs,
        "weight": weight,
        "reliability": rel,
        "supports": list(supports),
        "weakens": list(weakens),
        "panel": panel,
        "note": note,
    }


def build_atoms(mode, *, tokens, alphabet_class, unique, ic, ic_delta, norm_h,
                flat_chi, fp, letter_rendering, numeric_stats):
    """Return the list of evidence atoms for a stream under `mode`."""
    atoms = []
    lang_ic = LANG_IC_EN

    # ---- fingerprint priors (both modes) ----
    for key, susp in fp.suspicion_scores.items():
        fam = SUSPICION_TO_FAMILY.get(key)
        if fam is None:            # "unknown" and any unmapped -> emit nothing
            continue
        if susp <= 0:
            continue
        atoms.append(atom(f"fingerprint_prior:{fam}", 0.5 * susp, "low",
                          [fam], [], panel="fingerprint"))

    # In FIXED, the substitution-family frequency/shape atoms are NOT applicable
    # to a numeric integer-index stream (their interpretation presumes a
    # substitution alphabet — treating indices as symbols is the category error
    # the numeric_inconsistent atom records). Suppress them for numeric class.
    emit_sub_freq = not (mode == "FIXED" and alphabet_class == "numeric")

    # ---- frequency: IC atoms ----
    if emit_sub_freq and ic_delta is not None and abs(ic_delta) < IC_NEAR:
        atoms.append(atom("ic_near_language_reference", 0.25, "high",
                          ["monoalphabetic_substitution", "transposition"], [],
                          panel="frequency"))
    if emit_sub_freq and ic_delta is not None and ic_delta < -IC_DEPRESSED:
        atoms.append(atom("depressed_ic", 0.25, "high",
                          ["polyalphabetic_periodic", "homophonic_substitution"],
                          ["monoalphabetic_substitution", "transposition"],
                          panel="frequency"))

    # ---- frequency: shape atoms (mode-dependent thresholds) ----
    if mode == "ORIGINAL":
        peaked = (norm_h < 0.85) and (unique <= 28)
        flat = norm_h > 0.90
    else:  # FIXED — per-token chi-square flatness
        peaked = (flat_chi > PEAK_FLAT_MIN) and (unique <= PEAK_UNIQUE_MAX)
        flat = flat_chi < FLAT_FLAT_MAX
    if emit_sub_freq and peaked:
        atoms.append(atom("peaked_monogram_shape", 0.30, "high",
                          ["monoalphabetic_substitution", "transposition"],
                          ["homophonic_substitution"], panel="frequency"))
    if emit_sub_freq and flat:
        atoms.append(atom("flat_high_entropy", 0.30, "high",
                          ["homophonic_substitution"],
                          ["monoalphabetic_substitution"], panel="frequency"))

    # ---- shape: large symbol inventory ----
    if emit_sub_freq and unique > 26:
        w = min(0.45, 0.45 * (unique - 26) / 20.0)
        if w > 0:
            atoms.append(atom("large_symbol_inventory", w, "high",
                              ["homophonic_substitution"],
                              ["monoalphabetic_substitution"], panel="shape"))

    # ---- periodicity: periodic_ic_recovery (baselined) ----
    if mode == "ORIGINAL":
        # always-emit; the structural cols>=25 recovery test grades reliability
        # only (this is the reviewer finding-6 defect being reproduced).
        bpi = fp.best_period_ic
        bp = fp.best_period
        recovery_ok = False
        if bpi is not None and bp is not None and not math.isnan(ic):
            cols = len(tokens) // bp if bp else 0
            recovery_ok = (
                (bpi - ic) > PERIODIC_RECOVERY_MIN
                and bpi >= lang_ic - PERIODIC_ABS_MARGIN
                and cols >= PERIODIC_MIN_COLS
            )
        rel = "high" if recovery_ok else "low"
        atoms.append(atom("periodic_ic_recovery", 0.45, rel,
                          ["polyalphabetic_periodic"],
                          ["monoalphabetic_substitution", "transposition"],
                          panel="periodicity"))
    else:
        # FIXED — emit ONLY when a frequency-preserving shuffle null confirms
        # the recovery is significant (finding 6). Always high when emitted.
        sig, _p = periodic_recovery_significant(tokens, fp)
        if sig:
            atoms.append(atom("periodic_ic_recovery", 0.45, "high",
                              ["polyalphabetic_periodic"],
                              ["monoalphabetic_substitution", "transposition"],
                              panel="periodicity"))

    # ---- order_layout (needs letter_rendering) ----
    if letter_rendering is not None:
        mchi = monogram_chi2_vs_english(letter_rendering)
        qnll = normalized_ngram_score(letter_rendering, quad_logprobs(), 4)
        peaked_now = peaked
        if (mchi < MONO_CHI2_LOW) and (qnll < QUAD_NLL_SCRAMBLED):
            atoms.append(atom("letters_unsubstituted", 0.55, "high",
                              ["transposition"],
                              ["monoalphabetic_substitution",
                               "homophonic_substitution"], panel="order_layout"))
        if (mchi > MONO_CHI2_HIGH) and peaked_now:
            atoms.append(atom("letters_substituted", 0.35, "high",
                              ["monoalphabetic_substitution",
                               "homophonic_substitution"],
                              ["transposition"], panel="order_layout"))

    # ---- numeric battery ----
    if alphabet_class == "numeric" and numeric_stats is not None:
        ns = numeric_stats
        # numeric_token_stream: weakens list widens in FIXED (finding 1)
        if mode == "ORIGINAL":
            nts_weak = ["monoalphabetic_substitution", "homophonic_substitution",
                        "polyalphabetic_periodic", "transposition", "playfair"]
        else:
            nts_weak = list(SUBSTITUTION_PRIMARIES)
        atoms.append(atom("numeric_token_stream", 0.35, "high",
                          ["numeric_book_cipher", "nomenclator_codebook"],
                          nts_weak, panel="numeric_code"))

        # NEW in FIXED — scored numeric-vs-substitution inconsistency (finding 1)
        if mode == "FIXED" and ns["unique_rate"] > NUMERIC_UNIQUE_RATE:
            atoms.append(atom("numeric_inconsistent_with_substitution", 0.50,
                              "high", [], list(SUBSTITUTION_PRIMARIES),
                              panel="numeric_code",
                              note="unique/token %.3f > %.2f with numeric alphabet"
                              % (ns["unique_rate"], NUMERIC_UNIQUE_RATE)))

        # book_keylength_plausible: feasibility only (FIXED drops the hoax-weaken)
        if 1 <= ns["min"] and ns["max"] <= BOOK_KEYLEN_MAX:
            weak = ["plaintext_or_hoax"] if mode == "ORIGINAL" else []
            atoms.append(atom("book_keylength_plausible", 0.30, "high",
                              ["numeric_book_cipher"], weak, panel="numeric_code"))

        # front_loading_present (baselined:upper vs a uniform draw)
        fl = ns["front_loading_index"]
        if mode == "ORIGINAL":
            # always-emit when index>1; baseline grades reliability only
            rel = "high" if fl > 1.0 else "low"
            atoms.append(atom("front_loading_present", 0.20, rel,
                              ["numeric_book_cipher"], ["plaintext_or_hoax"],
                              panel="numeric_code"))
        else:
            # FIXED — emit ONLY when index>1 AND the parametric uniform null
            # rejects at p<=0.05 (finding 6). Always high when emitted.
            sig, _p = front_loading_significant(tokens, fl)
            if sig:
                atoms.append(atom("front_loading_present", 0.20, "high",
                                  ["numeric_book_cipher"], ["plaintext_or_hoax"],
                                  panel="numeric_code"))

        # independent_random_like (k-of-n structural); weight 0.36 in FIXED
        cond = [
            ns["last_digit_uniform"],
            ns["benford_eps"] >= BENFORD_INCONSISTENT,
            ns["repeat_within_1sd"],
        ]
        if sum(1 for c in cond if c) >= 2:
            w = 0.40 if mode == "ORIGINAL" else 0.36
            atoms.append(atom("independent_random_like", w, "high",
                              ["plaintext_or_hoax"],
                              ["numeric_book_cipher", "nomenclator_codebook"],
                              panel="numeric_code"))

        # structured_hoax_artifact (baselined): last-DIGIT preference (finding 8)
        hcond = [
            ns["monotone_run_excess_p"] <= 0.05,
            (not ns["last_digit_uniform"]),         # last-digit chi2 significant
            False,  # consecutive-run excess: not synthesised here (neutral)
        ]
        if sum(1 for c in hcond if c) >= 2:
            atoms.append(atom("structured_hoax_artifact", 0.35, "high",
                              ["plaintext_or_hoax"], ["numeric_book_cipher"],
                              panel="numeric_code"))

        # subtype atoms (finding 7) — P8 word/char/skip hypothesis flags
        if mode == "FIXED":
            book_feasible = ns["max"] <= BOOK_KEYLEN_MAX
            front = fl >= FRONTLOAD_MIN
            below_rep = ns["repeat_rate"] < ns["repeat_null_mean"]
            if book_feasible and (front or below_rep):
                atoms.append(atom("book_word_position_signature", 0.40, "high",
                                  ["numeric_word_position"], [],
                                  panel="numeric_code"))

    return atoms


# =====================================================================
# Scoring / verdict
# =====================================================================
def rel_mult(rel):
    return 1.0 if rel == "high" else 0.4


def score_families(atoms):
    fams = set(PRIMARIES) | set(SUBTYPES)
    raw = {f: 0.0 for f in fams}
    ev = {f: [] for f in fams}
    counter_ev = {f: [] for f in fams}
    high_support = {f: 0 for f in fams}
    for a in atoms:
        w = a["weight"] * rel_mult(a["reliability"])
        for f in a["supports"]:
            if f in raw:
                raw[f] += w
                ev[f].append(a["observation"])
                if a["reliability"] == "high":
                    high_support[f] += 1
        for f in a["weakens"]:
            if f in raw:
                raw[f] -= w
                counter_ev[f].append(a["observation"])
    # Round to 6 decimals to match production _score_families (diagnosis.py):
    # clears float-summation noise (0.35+0.30+0.20-0.35 -> 0.4999999999999999) so
    # the strict margin gate is stable at exact boundaries (beale_3 margin is
    # exactly 0.15 -> confident under the strict `< 0.15` rule).
    score = {f: round(max(0.0, min(1.0, raw[f])), 6) for f in fams}
    return score, ev, counter_ev, high_support


def discriminator_status(did, panel_status):
    splits, dep, st = DISCRIMINATORS[did]
    if st == "planned":
        return "unavailable"
    if all(panel_status.get(p) == "ok" for p in dep):
        return "run"
    return "pending"


def confidence(fam, score, high_support):
    s = score[fam]
    conf_set = CONFUSABLE.get(fam, set()) & set(PRIMARIES)
    margin = s - (max((score[c] for c in conf_set), default=0.0))
    if s >= 0.70 and high_support[fam] >= 2 and margin >= 0.25:
        return "strong", margin
    if s >= 0.45 and high_support[fam] >= 1:
        return "moderate", margin
    if s > 0:
        return "weak", margin
    return "none", margin


def diagnose(atoms, token_count, panel_status):
    score, ev, counter_ev, high_support = score_families(atoms)
    ranked = sorted(PRIMARIES, key=lambda f: (-score[f], f))
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None

    uncertain = False
    reasons = []
    # (a) token gate
    if token_count < 60:
        uncertain = True
        reasons.append("a:token_count<60")
    # (b) margin
    if top2 is not None and (score[top1] - score[top2]) < 0.15:
        uncertain = True
        reasons.append("b:margin<0.15")
    # (c) top1 has discriminators and every one pending
    discs = FAMILY_DISCS.get(top1, [])
    if discs and all(
        discriminator_status(d, panel_status) == "pending" for d in discs
    ):
        uncertain = True
        reasons.append("c:all_pending")
    # (d) mutually confusable top-2 with no run discriminator splitting them
    if top2 is not None and top2 in CONFUSABLE.get(top1, set()):
        covering = [
            d for d, (sp, _, _) in DISCRIMINATORS.items()
            if sp == frozenset({top1, top2})
        ]
        if covering and all(
            discriminator_status(d, panel_status) != "run" for d in covering
        ):
            uncertain = True
            reasons.append("d:confusable_disc_not_run")

    verdict = "uncertain" if uncertain else "confident"

    # recommended_next: prefer a discriminator whose splits == {top1, top2};
    # FALLBACK (finding 5, when no discriminator covers the actual top-2):
    # recommend a discriminator that NAMES top2 (the challenger) and is
    # actionable now (all depended panels ok / status run), else one naming
    # top1, else any naming either. This surfaces the panel that best separates
    # the pair (e.g. homophonic-vs-poly -> disc_sub_periodic exercises the
    # periodicity test that resolves it).
    rec = None
    if uncertain and top2 is not None:
        cover = [
            d for d, (sp, _, _) in DISCRIMINATORS.items()
            if sp == frozenset({top1, top2})
        ]
        if cover:
            rec = cover[0]
        else:
            def _actionable(d):
                return discriminator_status(d, panel_status) == "run"
            for fam in (top2, top1):
                cands = FAMILY_DISCS.get(fam, [])
                runnable = [d for d in cands if _actionable(d)]
                pool = runnable or cands
                if pool:
                    rec = pool[0]
                    break

    conf, margin = confidence(top1, score, high_support)
    return {
        "score": score,
        "ranked": ranked,
        "top1": top1,
        "top2": top2,
        "verdict": verdict,
        "reasons": reasons,
        "recommended_next": rec,
        "top1_confidence": conf,
        "top1_margin": margin,
        "evidence": ev,
        "counterevidence": counter_ev,
        "high_support": high_support,
    }


# =====================================================================
# Full stream diagnosis (build the inputs the atoms need)
# =====================================================================
def run_stream(mode, tokens, alphabet_class, letter_rendering=None,
               word_group_count=0):
    n = len(tokens)
    unique = len(set(tokens))
    fp = compute_cipher_fingerprint(tokens, max(unique, 26), language="en",
                                    word_group_count=word_group_count)
    ic = fp.ic
    ic_delta = fp.ic_delta_from_reference
    counts = Counter(tokens)
    norm_h = _normalized_entropy(counts, n)
    flat_chi = _chi2_vs_uniform(counts, n) / n if n else 0.0

    numeric_stats = numeric_battery(tokens) if alphabet_class == "numeric" else None

    atoms = build_atoms(mode, tokens=tokens, alphabet_class=alphabet_class,
                        unique=unique, ic=ic, ic_delta=ic_delta, norm_h=norm_h,
                        flat_chi=flat_chi, fp=fp, letter_rendering=letter_rendering,
                        numeric_stats=numeric_stats)

    # panel_status for discriminator predicate
    panel_status = {
        "frequency": "ok", "shape": "ok", "periodicity": "ok",
        "polygraphic": "ok", "numeric_code": "ok" if alphabet_class == "numeric"
        else "not_computable",
        "order_layout": "ok" if letter_rendering is not None else "not_computable",
    }
    result = diagnose(atoms, n, panel_status)
    result["_diag"] = {
        "n": n, "unique": unique, "ic": ic, "ic_delta": ic_delta,
        "norm_h": norm_h, "flat_chi": flat_chi,
    }
    result["_atoms"] = atoms
    return result


# =====================================================================
# Fixture definitions (Part 9)
# =====================================================================
def build_fixtures():
    """Return dict name -> (mode-independent stream inputs, expected)."""
    fx = {}

    # ---- confident set ----
    fx["A_mono"] = dict(
        tokens=None, alphabet_class="letters",
        gen=lambda: gen_mono(load_plain(74)[:300], 11),
        expected_top1="monoalphabetic_substitution", expected_verdict="confident",
    )
    fx["B_periodic"] = dict(
        alphabet_class="letters",
        gen=lambda: gen_vigenere(load_plain(76)[:300], 5, 5),
        expected_top1="polyalphabetic_periodic", expected_verdict="confident",
    )
    fx["C_homophonic"] = dict(
        alphabet_class="symbols",
        gen=lambda: gen_homophonic(load_plain(11)[:300], 52, 3),
        expected_top1="homophonic_substitution", expected_verdict="confident",
    )
    fx["D_transposition"] = dict(
        alphabet_class="letters",
        gen=lambda: gen_columnar(load_plain(84)[:400], 9, 5),
        expected_top1="transposition", expected_verdict="confident",
    )
    fx["E_numeric_book"] = dict(
        alphabet_class="numeric",
        gen=lambda: (gen_book_word_position(300, 1320, 7), None),
        expected_top1="numeric_book_cipher", expected_verdict="confident",
    )

    # ---- near-miss set ----
    fx["i_short_mono"] = dict(
        alphabet_class="letters",
        gen=lambda: gen_mono(load_plain(74)[:45], 7),
        expected_top1="monoalphabetic_substitution", expected_verdict="uncertain",
        expected_reason="a",
    )
    fx["ii_light_homophonic"] = dict(
        alphabet_class="mixed",
        # light homophonic: ~3 extra homophones -> mild IC depression (NOT flat,
        # small inventory). Depressed IC makes it genuinely confusable with a
        # short-period polyalphabetic (both fire depressed_ic); homophonic leads
        # narrowly via the inventory/prior. Robust uncertain across pages/seeds.
        gen=lambda: gen_homophonic(load_plain(98)[:150], 29, 4),
        expected_top1="homophonic_substitution",
        expected_top2="polyalphabetic_periodic", expected_verdict="uncertain",
        expected_reason="b", expected_rec="disc_sub_periodic",
    )
    fx["iii_transp_norender"] = dict(
        alphabet_class="symbols",
        gen=lambda: gen_columnar(load_plain(84)[:120], 9, 6),
        expected_top1="monoalphabetic_substitution", expected_verdict="uncertain",
        expected_reason="d", withhold_rendering=True,
    )
    fx["iv_uniform_random"] = dict(
        alphabet_class="numeric",
        gen=lambda: (gen_uniform(300, 1, 3000, 42), None),
        expected_top1="plaintext_or_hoax", expected_verdict="uncertain",
        expected_reason="b",
    )
    return fx


def eval_fixtures(mode):
    fx = build_fixtures()
    rows = []
    for name, spec in fx.items():
        ids, rend = spec["gen"]()
        alphabet_class = spec["alphabet_class"]
        if alphabet_class == "letters" and not spec.get("withhold_rendering"):
            letter_rendering = rend
        else:
            letter_rendering = None
        res = run_stream(mode, ids, alphabet_class, letter_rendering=letter_rendering)
        rows.append((name, spec, res))
    return rows


# =====================================================================
# Part 5 — island span rule (candidate junction rules + Monte-Carlo)
# =====================================================================
def build_function_words(top_k=150):
    """Closed-class proxy: the most frequent dictionary words (english_common is
    frequency-ordered; INV-0 ships a curated resources/function_words/en.txt)."""
    with open("resources/dictionaries/english_common.txt") as f:
        words = [w.strip() for w in f if w.strip()]
    return set(words[:top_k])


def build_dict_set():
    with open("resources/dictionaries/english_common.txt") as f:
        return set(w.strip() for w in f if w.strip())


def build_attested_bigrams(top_k=4000):
    words = []
    for p in [1, 2, 5, 7, 10, 15]:
        words += load_prose_words(p, skip_words=200, n_words=8000)
    bg = Counter(zip(words, words[1:]))
    common = dict(bg.most_common(top_k))
    ctot = sum(common.values())
    floor = math.log10(0.1 / ctot)
    lp = {pair: math.log10(c / ctot) for pair, c in common.items()}
    return set(common), lp, floor


def word_bigram_loglik(words, lp, floor):
    if len(words) < 2:
        return floor
    return sum(lp.get((words[k], words[k + 1]), floor)
               for k in range(len(words) - 1)) / (len(words) - 1)


def word_bigram_order_p(words, lp, floor, seed=0, n_shuffles=300):
    """Upper-tail shuffle-null p that observed word-bigram loglik exceeds a
    frequency-preserving permutation (the finding-3 order-sensitivity test)."""
    obs = word_bigram_loglik(words, lp, floor)
    rng = random.Random(seed)
    pool = list(words)
    ge = 0
    for _ in range(n_shuffles):
        rng.shuffle(pool)
        if word_bigram_loglik(pool, lp, floor) >= obs:
            ge += 1
    return (ge + 1) / (n_shuffles + 1)


ATTESTED_PER_SPAN = 2   # finding-2 option-2 attested-junction floor


def longest_coherent_span(words, dict_set, function_words, attested,
                          min_attested=ATTESTED_PER_SPAN):
    """Longest run in which every junction is adjacent_ok AND the run as a whole
    contains >= `min_attested` attested word-bigram junctions (finding 2,
    option 2). adjacent_ok = BOTH words in the dictionary AND (attested_bigram
    OR a function word is involved). The function-word rescue keeps content
    words glued to their neighbours (real prose), but a run only *qualifies* as
    coherent when it also carries enough attested high-frequency bigrams — which
    shuffled prose (function-word-dense but attestation-sparse) does not."""
    if len(words) < 2:
        return len(words)

    def ok(k):
        a, b = words[k], words[k + 1]
        if a not in dict_set or b not in dict_set:
            return False, False
        att = (a, b) in attested
        adj = att or a in function_words or b in function_words
        return adj, att

    best = 1
    start = 0
    att_in_run = 0
    run_len = 1
    for k in range(len(words) - 1):
        adj, att = ok(k)
        if adj:
            run_len += 1
            att_in_run += 1 if att else 0
            if att_in_run >= min_attested:
                best = max(best, run_len)
        else:
            run_len = 1
            att_in_run = 0
    return best


B2_PLAINTEXT = (
    "I HAVE DEPOSITED IN THE COUNTY OF BEDFORD ABOUT FOUR MILES FROM BUFORDS "
    "IN AN EXCAVATION OR VAULT SIX FEET BELOW THE SURFACE OF THE GROUND THE "
    "FOLLOWING ARTICLES BELONGING JOINTLY TO THE PARTIES WHOSE NAMES ARE GIVEN "
    "IN NUMBER THREE HEREWITH THE FIRST DEPOSIT CONSISTED OF ONE THOUSAND AND "
    "FOURTEEN POUNDS OF GOLD AND THREE THOUSAND EIGHT HUNDRED AND TWELVE POUNDS "
    "OF SILVER DEPOSITED NOVEMBER EIGHTEEN NINETEEN"
).split()


def island_verdict(words, *, lang, dict_set, function_words, attested, lp, floor,
                   has_bigram_resource, shuffle_seed=0):
    n = len(words)
    dict_rate = sum(1 for w in words if w in dict_set) / n if n else 0.0
    fn_rate = sum(1 for w in words if w in function_words) / n if n else 0.0
    span = longest_coherent_span(words, dict_set, function_words, attested)
    order_p = (word_bigram_order_p(words, lp, floor, seed=shuffle_seed)
               if has_bigram_resource else 1.0)
    order_sig = order_p <= 0.05
    fn_min = 0.25 if lang == "en" else 0.10   # la/de are function-word-poorer
    coherent = (dict_rate >= 0.75 and fn_rate >= fn_min and span >= 5
                and has_bigram_resource and order_sig)
    if coherent:
        verdict = "coherent"
    elif dict_rate < 0.35 or (fn_rate < 0.08 and span <= 2):
        verdict = "gibberish"
    else:
        verdict = "word_islands"
    return {
        "verdict": verdict, "dict_rate": dict_rate, "fn_rate": fn_rate,
        "span": span, "order_p": order_p, "order_sig": order_sig,
        "word_bigram_available": has_bigram_resource,
    }


def island_calibration():
    fnw = build_function_words()
    dset = build_dict_set()
    att, lp, floor = build_attested_bigrams()
    print("\n" + "=" * 72)
    print("PART 5 — island_report verdict (finding 2/3: shuffled != coherent)")
    print("=" * 72)
    print(f"function_words={len(fnw)}  dict_words={len(dset)}  "
          f"attested_bigrams={len(att)}")

    def vd(words, lang="en", has_res=True, seed=0):
        return island_verdict(words, lang=lang, dict_set=dset, function_words=fnw,
                              attested=att, lp=lp, floor=floor,
                              has_bigram_resource=has_res, shuffle_seed=seed)

    ordered = [load_prose_words(p, skip_words=500, n_words=260) for p in CORP[:6]]

    print("\nOrdered English prose (want coherent):")
    ord_coh = 0
    for i, w in enumerate(ordered):
        r = vd(w)
        ord_coh += (r["verdict"] == "coherent")
        print(f"  s{i}: {r['verdict']:12s} dict={r['dict_rate']:.2f} "
              f"fn={r['fn_rate']:.2f} span={r['span']} order_p={r['order_p']:.4f}")

    print("\nShuffled English prose (want != coherent) — Monte-Carlo:")
    rng = random.Random(123)
    false_coh = 0
    total = 0
    for w in ordered:
        for _ in range(40):
            ws = list(w)
            rng.shuffle(ws)
            r = vd(ws, seed=7)
            total += 1
            false_coh += (r["verdict"] == "coherent")
    print(f"  falsely-coherent shuffles: {false_coh}/{total} "
          f"(OR-rescue baseline was 240/240; nominal alpha floor ~5%)")

    # a single fixed-seed shuffled fixture (the P9 test uses one shuffle)
    fix_shuf = list(ordered[4])
    random.Random(2024).shuffle(fix_shuf)
    rshuf = vd(fix_shuf, seed=11)
    print(f"  fixed shuffled fixture (seed 2024): {rshuf['verdict']} "
          f"span={rshuf['span']} order_p={rshuf['order_p']:.4f}")

    rb2 = vd(B2_PLAINTEXT)
    print(f"\nB2 self-check plaintext: {rb2['verdict']} dict={rb2['dict_rate']:.2f} "
          f"fn={rb2['fn_rate']:.2f} span={rb2['span']} order_p={rb2['order_p']:.4f} "
          f"(want coherent)")

    rla = vd(B2_PLAINTEXT, lang="la", has_res=False)
    print(f"la (no bigram resource): {rla['verdict']} "
          f"(want != coherent; word_bigram_available={rla['word_bigram_available']})")

    coherent_prose_ok = ord_coh >= 1
    shuffled_ok = rshuf["verdict"] != "coherent"
    b2_ok = rb2["verdict"] == "coherent"
    la_ok = rla["verdict"] != "coherent"
    print(f"\nPart-5 checks: coherent_prose={coherent_prose_ok} "
          f"shuffled_fixture!=coherent={shuffled_ok} B2_coherent={b2_ok} "
          f"la!=coherent={la_ok}")
    return all([coherent_prose_ok, shuffled_ok, b2_ok, la_ok])


# =====================================================================
# Reporting
# =====================================================================
DISP = {
    "monoalphabetic_substitution": "mono",
    "homophonic_substitution": "homophonic",
    "polyalphabetic_periodic": "polyalphabetic",
    "transposition": "transposition",
    "transposition_homophonic": "transp_homo",
    "numeric_book_cipher": "numeric_book",
    "plaintext_or_hoax": "plaintext_or_hoax",
    "nomenclator_codebook": "nomenclator",
    "playfair": "playfair",
}


def short(f):
    return DISP.get(f, f)


def fixture_table(mode):
    print("\n" + "=" * 72)
    print(f"FIXTURE EVALUATION — {mode} catalog")
    print("=" * 72)
    rows = eval_fixtures(mode)
    all_pass = True
    for name, spec, res in rows:
        top1, top2 = res["top1"], res["top2"]
        s1 = res["score"][top1]
        s2 = res["score"][top2] if top2 else 0.0
        want_top1 = spec["expected_top1"]
        want_verd = spec["expected_verdict"]
        ok_top1 = (top1 == want_top1)
        ok_top2 = ("expected_top2" not in spec) or (top2 == spec["expected_top2"])
        ok_verd = (res["verdict"] == want_verd)
        strong_ok = True
        if want_verd == "confident":
            strong_ok = (res["top1_confidence"] == "strong")
        rec_ok = True
        if want_verd == "uncertain":
            # recommended_next must address the actual top-2 (cover it, or the
            # defined fallback: name one of the two)
            rec = res["recommended_next"]
            want_rec = spec.get("expected_rec")
            if want_rec is not None:
                rec_ok = (rec == want_rec)
            elif rec is not None:
                sp = DISCRIMINATORS[rec][0]
                rec_ok = (sp == frozenset({top1, top2})
                          or top1 in sp or top2 in sp)
        row_pass = ok_top1 and ok_top2 and ok_verd and strong_ok and rec_ok
        all_pass = all_pass and row_pass
        flag = "PASS" if row_pass else "**FAIL**"
        print(f"\n[{flag}] {name}")
        print(f"   top1={short(top1)}({s1:.3f}) top2={short(top2)}({s2:.3f}) "
              f"margin={s1 - s2:.3f}")
        print(f"   verdict={res['verdict']} reasons={res['reasons']} "
              f"conf={res['top1_confidence']} rec_next={res['recommended_next']}")
        print(f"   evidence[{short(top1)}]={res['evidence'][top1]}")
        if not ok_top1:
            print(f"   !! expected top1={short(want_top1)}")
        if not ok_verd:
            print(f"   !! expected verdict={want_verd}")
        if want_verd == "confident" and not strong_ok:
            print(f"   !! expected top1 confidence=strong got {res['top1_confidence']}")
    print(f"\n{mode}: fixtures {'ALL PASS' if all_pass else 'HAVE FAILURES'}")
    return all_pass, [(n, r) for (n, _s, r) in rows]


def beale_acceptance(mode):
    print("\n" + "=" * 72)
    print(f"ACCEPTANCE 2 — beale_1 / beale_3 ranked tables — {mode} catalog")
    print("=" * 72)
    ok = True
    for name in ("beale_1", "beale_3"):
        vals = load_beale(name)
        res = run_stream(mode, vals, "numeric")
        score = res["score"]
        nb = score["numeric_book_cipher"]
        sub_max = max(score[f] for f in SUBSTITUTION_PRIMARIES)
        worst_sub = max(SUBSTITUTION_PRIMARIES, key=lambda f: score[f])
        passes = nb > sub_max
        ok = ok and passes
        print(f"\n{name}: numeric_book={nb:.3f}  "
              f"max_substitution={sub_max:.3f} ({short(worst_sub)})  "
              f"{'PASS' if passes else '**FAIL**'}")
        ranked = sorted(PRIMARIES, key=lambda f: (-score[f], f))
        for f in ranked[:6]:
            if score[f] > 0:
                print(f"     {short(f):18s} {score[f]:.3f}")
    print(f"\n{mode}: acceptance-2 {'PASS' if ok else '**FAIL**'}")
    return ok


def reproduce_reviewer_failures():
    """Confirm the second-review failures on real data BEFORE applying fixes."""
    print("\n" + "=" * 72)
    print("REPRODUCTION — second-review failures on real cipher_id.py + Beale")
    print("=" * 72)

    # (R2-3) English letter streams measure norm_entropy ~0.90 -> v1 peaked
    # (<0.85) never fires, flat (>0.90) coin-flips.
    nes = []
    for p in CORP[:6]:
        ids, _ = gen_mono(load_plain(p)[:300], 0)
        nes.append(_normalized_entropy(Counter(ids), len(ids)))
    print(f"R2-3 English mono norm_entropy: min={min(nes):.4f} max={max(nes):.4f} "
          f"(v1 peaked<0.85 never fires; flat>0.90 misfires)")

    # (R2-1) beale_1 homophonic ties/beats numeric_book under ORIGINAL catalog.
    for name in ("beale_1", "beale_3"):
        vals = load_beale(name)
        res = run_stream("ORIGINAL", vals, "numeric")
        sc = res["score"]
        print(f"R2-1 {name} ORIGINAL: homophonic={sc['homophonic_substitution']:.3f} "
              f"numeric_book={sc['numeric_book_cipher']:.3f} top1={short(res['top1'])} "
              f"(homophonic ties/beats numeric_book -> acceptance-2 unpassable)")

    # (R2-2) OR-rescue span rule reads shuffled prose as coherent (spans >= 5).
    fnw = build_function_words()
    dset = build_dict_set()
    att, lp, floor = build_attested_bigrams()
    w = load_prose_words(CORP[4], skip_words=500, n_words=260)
    rng = random.Random(0)
    ge5 = 0
    for _ in range(40):
        ws = list(w)
        rng.shuffle(ws)
        best = cur = 1
        for k in range(len(ws) - 1):
            a, b = ws[k], ws[k + 1]
            ok = ((a, b) in att or a in fnw or b in fnw) and a in dset and b in dset
            cur = cur + 1 if ok else 1
            best = max(best, cur)
        ge5 += best >= 5
    print(f"R2-2 OR-rescue span on shuffled prose: {ge5}/40 shuffles span>=5 "
          f"(shuffled reads coherent -> finding-3 unpassable)")


def main():
    print("#" * 72)
    print("# INV-0 scoring calibration  (real cipher_id.py + real Beale data)")
    print("#" * 72)

    reproduce_reviewer_failures()

    results = {}
    for mode in ("ORIGINAL", "FIXED"):
        fp, rows = fixture_table(mode)
        bp = beale_acceptance(mode)
        results[mode] = {"fixtures_pass": fp, "beale_pass": bp,
                         "rows": dict(rows)}

    island_ok = island_calibration()

    # ---- before/after fixture score table ----
    print("\n" + "#" * 72)
    print("# BEFORE / AFTER fixture score table (top1 family : score : verdict)")
    print("#" * 72)
    fx = build_fixtures()
    hdr = f"{'fixture':22s} {'ORIGINAL':34s} {'FIXED':34s}"
    print(hdr)
    print("-" * len(hdr))
    for name in fx:
        o = results["ORIGINAL"]["rows"][name]
        f = results["FIXED"]["rows"][name]
        os_ = f"{short(o['top1'])}={o['score'][o['top1']]:.2f} {o['verdict']}"
        fs_ = f"{short(f['top1'])}={f['score'][f['top1']]:.2f} {f['verdict']} " \
              f"[{f['top1_confidence']}]"
        want = fx[name]["expected_top1"]
        wv = fx[name]["expected_verdict"]
        ok = (f["top1"] == want and f["verdict"] == wv)
        print(f"{name:22s} {os_:34s} {fs_:34s} {'OK' if ok else 'XX'}")

    print("\n" + "#" * 72)
    print("# SUMMARY")
    print("#" * 72)
    print(f"ORIGINAL: fixtures_pass={results['ORIGINAL']['fixtures_pass']} "
          f"acceptance2_pass={results['ORIGINAL']['beale_pass']}  "
          f"(reproduces reviewer failures)")
    print(f"FIXED   : fixtures_pass={results['FIXED']['fixtures_pass']} "
          f"acceptance2_pass={results['FIXED']['beale_pass']} "
          f"part5_island_pass={island_ok}")
    ok = (results["FIXED"]["fixtures_pass"] and results["FIXED"]["beale_pass"]
          and island_ok)
    print(f"\nFIXED catalog: {'ALL CHECKS PASS' if ok else 'FAILURES REMAIN'}")


if __name__ == "__main__":
    main()
