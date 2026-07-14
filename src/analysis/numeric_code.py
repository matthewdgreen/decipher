"""P8 numeric battery for numeric-code / book-cipher diagnosis.

INV-0 Part 4. Beale-family ciphers are streams of decimal integers indexing
into a key text (a book). Substitution statistics are the wrong lens for them
(treating a value like 2906 as "symbol #2906" is a category error). This module
provides the numeric-native battery: digit laws (Benford / uniform), Gillogly
run-artifact detectors, book-key-length feasibility, front-loading skew, and a
split set of hypothesis flags. All order-sensitive statistics are baselined
against a null via :mod:`analysis.null_baseline`.

The battery is consumed by :mod:`analysis.panels` (the ``numeric_code`` panel)
and by ``scripts/research/beale_report.py``. It never sees plaintext or any key
text; ``related_profile`` may only carry another *ciphertext's* own numeric
measurements (Beale 2), never ground truth.
"""
from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, Sequence

from analysis.null_baseline import null_percentile, parametric_percentile

__all__ = [
    "parse_numeric_ciphertext",
    "is_numeric_ciphertext",
    "numeric_code_battery",
    "alphabetical_run_report",
    "profile_for_related",
    "assert_not_solution_bearing",
    "is_solution_bearing",
    "BENFORD_INCONSISTENT_EPSILON",
    "BOOK_KEYLEN_MAX",
    "FRONTLOAD_MIN",
    "SOLUTION_BEARING_MARKER",
    "BEALE_DOI_KEY_PATH",
]

# --- Solution-bearing key-text guard (finding 9) -------------------------------
# The Declaration-of-Independence Beale key is a DECRYPTION KEY, never diagnostic
# input. The resource carries this marker in its banner; any file-loading path
# that feeds the diagnosis layer must refuse a file bearing it.
SOLUTION_BEARING_MARKER = "SOLUTION-BEARING KEY TEXT"
BEALE_DOI_KEY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "resources", "reference", "beale_doi_key_words.txt",
)


def is_solution_bearing(content: str) -> bool:
    """True iff ``content`` is a marked solution-bearing key text."""
    return SOLUTION_BEARING_MARKER in content


def assert_not_solution_bearing(content: str, *, source: str = "") -> None:
    """Raise ``ValueError`` if ``content`` is a solution-bearing key text.

    Called by every path that would feed a file to the diagnosis layer, so a
    decryption key can never be diagnosed as if it were ciphertext (finding 9).
    """
    if is_solution_bearing(content):
        where = f" ({source})" if source else ""
        raise ValueError(
            f"refusing to load solution-bearing key text as diagnostic input{where}"
        )

# Derived constants (kept in sync with calibrate_inv0_scoring.py FIXED mode).
BENFORD_INCONSISTENT_EPSILON = 0.12   # epsilon_benford >= this -> random-like digit law
BOOK_KEYLEN_MAX = 100_000             # max(values) <= this -> plausible book key length
FRONTLOAD_MIN = 1.5                   # front_loading_index >= this -> structural front skew
# chi-square 0.05 critical value, 9 degrees of freedom (first-digit 1..9, last-digit 0..9).
_CHI2_9DOF_05 = 16.919
# Companion-sharing thresholds (Beale 2 as related_profile; not load-bearing for scoring).
_SHARED_JACCARD_MIN = 0.30
_SHARED_CORR_MIN = 0.30


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_numeric_ciphertext(text: str) -> list[int]:
    """Parse whitespace/comma-separated decimal tokens into a list of ints.

    Raises ``ValueError`` naming the first token that is not a decimal integer.
    """
    raw = text.replace(",", " ").split()
    values: list[int] = []
    for tok in raw:
        try:
            values.append(int(tok))
        except ValueError:
            raise ValueError(f"non-numeric token in numeric ciphertext: {tok!r}") from None
    return values


def is_numeric_ciphertext(text: str) -> bool:
    """True iff ``text`` is a non-empty stream of decimal (optionally comma-sep) tokens."""
    raw = text.replace(",", " ").split()
    if not raw:
        return False
    for tok in raw:
        if not (tok.isdigit() or (tok[0] in "+-" and tok[1:].isdigit())):
            return False
    return True


# ---------------------------------------------------------------------------
# Digit / value helpers
# ---------------------------------------------------------------------------

def _first_digit(v: int) -> int:
    return int(str(abs(v))[0])


def _last_digit(v: int) -> int:
    return abs(v) % 10


def _benford_probs() -> dict[int, float]:
    return {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def benford_epsilon(values: Sequence[int]) -> float:
    """sup-norm distance of the empirical first-digit law from Benford (Wase 2021)."""
    n = len(values)
    if not n:
        return 0.0
    fd = Counter(_first_digit(v) for v in values)
    obs = {d: fd.get(d, 0) / n for d in range(1, 10)}
    ben = _benford_probs()
    return max(abs(obs[d] - ben[d]) for d in range(1, 10))


def _total_variation_benford(values: Sequence[int]) -> float:
    n = len(values)
    if not n:
        return 0.0
    fd = Counter(_first_digit(v) for v in values)
    obs = {d: fd.get(d, 0) / n for d in range(1, 10)}
    ben = _benford_probs()
    return 0.5 * sum(abs(obs[d] - ben[d]) for d in range(1, 10))


def _first_digit_chi2_uniform(values: Sequence[int]) -> float:
    n = len(values)
    if not n:
        return 0.0
    fd = Counter(_first_digit(v) for v in values)
    exp = n / 9.0
    return sum((fd.get(d, 0) - exp) ** 2 / exp for d in range(1, 10))


def _first_digit_chi2_benford(values: Sequence[int]) -> float:
    n = len(values)
    if not n:
        return 0.0
    fd = Counter(_first_digit(v) for v in values)
    ben = _benford_probs()
    total = 0.0
    for d in range(1, 10):
        exp = ben[d] * n
        if exp > 0:
            total += (fd.get(d, 0) - exp) ** 2 / exp
    return total


def last_digit_chi2(values: Sequence[int]) -> float:
    n = len(values)
    if not n:
        return 0.0
    ld = Counter(_last_digit(v) for v in values)
    exp = n / 10.0
    return sum((ld.get(d, 0) - exp) ** 2 / exp for d in range(10))


def front_loading_index(values: Sequence[int]) -> float:
    """(share of values <= max/5) / 0.2. >1 means book-order front skew."""
    if not values:
        return 0.0
    mx = max(values)
    if mx <= 0:
        return 0.0
    share = sum(1 for v in values if v <= mx / 5.0) / len(values)
    return share / 0.2


def repeat_rate(values: Sequence[int]) -> float:
    n = len(values)
    if not n:
        return 0.0
    c = Counter(values)
    return sum(cnt for cnt in c.values() if cnt >= 2) / n


def longest_monotone_run(values: Sequence[int]) -> int:
    """Longest non-decreasing run length."""
    if not values:
        return 0
    best = cur = 1
    for i in range(1, len(values)):
        if values[i] >= values[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def longest_strictly_increasing_run(values: Sequence[int]) -> int:
    if not values:
        return 0
    best = cur = 1
    for i in range(1, len(values)):
        if values[i] > values[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def consecutive_integer_pairs(values: Sequence[int]) -> int:
    """Count of adjacent pairs whose second value is exactly one more than the first."""
    return sum(1 for i in range(len(values) - 1) if values[i + 1] - values[i] == 1)


def _repeated_ngram_count(values: Sequence[int], n: int) -> int:
    """Number of length-n windows whose n-gram occurs at least twice."""
    if len(values) < n:
        return 0
    grams = Counter(tuple(values[i : i + n]) for i in range(len(values) - n + 1))
    return sum(c for c in grams.values() if c >= 2)


def _repeated_ngram_positions(values: Sequence[int], n: int, top: int = 20) -> list[dict[str, Any]]:
    if len(values) < n:
        return []
    positions: dict[tuple, list[int]] = {}
    for i in range(len(values) - n + 1):
        positions.setdefault(tuple(values[i : i + n]), []).append(i)
    out = [
        {"ngram": list(g), "count": len(p), "positions": p[:10]}
        for g, p in positions.items()
        if len(p) >= 2
    ]
    out.sort(key=lambda r: (-r["count"], r["ngram"]))
    return out[:top]


def _uniform_sampler(lo: int, hi: int):
    def sample(rng, n: int) -> list[int]:
        return [rng.randint(lo, hi) for _ in range(n)]
    return sample


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------

def numeric_code_battery(
    values: Sequence[int],
    *,
    related_profile: dict[str, Any] | None = None,
    rng_namespace: str = "",
    n_shuffles: int = 1000,
    include_modulo: bool = True,
    include_repeat_baselines: bool = True,
) -> dict[str, Any]:
    """Full P8 numeric battery over ``values`` (Part 4 groups).

    ``include_modulo`` / ``include_repeat_baselines`` gate the expensive null
    baselines that the diagnosis atom layer does not read (the modulo chi-square
    percentiles and the repeated-n-gram excess baselines). The ``diagnose`` hot
    path turns them off; ``beale_report``/CLI leave them on for the full report.
    """
    values = list(values)
    n = len(values)
    if n == 0:
        return {"status": "not_computable", "reason": "empty", "count": 0}

    uniq = sorted(set(values))
    counts = Counter(values)
    mx = max(values)
    mn = min(values)

    # ---- basic ----
    sorted_uniq = uniq
    gaps = [sorted_uniq[i + 1] - sorted_uniq[i] for i in range(len(sorted_uniq) - 1)]
    gap_hist = dict(Counter(gaps))
    basic = {
        "count": n,
        "unique": len(uniq),
        "unique_rate": len(uniq) / n,
        "min": mn,
        "max": mx,
        "repeated_token_rate": repeat_rate(values),
        "top10": [{"value": v, "count": c} for v, c in counts.most_common(10)],
        "gap_histogram": {str(k): v for k, v in sorted(gap_hist.items())},
        "max_gap": max(gaps) if gaps else 0,
        "gaps_equal_one": gap_hist.get(1, 0),
    }

    # ---- digits + benford ----
    ld_chi2 = last_digit_chi2(values)
    digits = {
        "first_digit_dist": {str(d): Counter(_first_digit(v) for v in values).get(d, 0) for d in range(1, 10)},
        "last_digit_dist": {str(d): Counter(_last_digit(v) for v in values).get(d, 0) for d in range(10)},
        "first_digit_chi2_uniform": _first_digit_chi2_uniform(values),
        "last_digit_chi2_uniform": ld_chi2,
        "last_digit_uniform": ld_chi2 < _CHI2_9DOF_05,
    }
    benford = {
        "first_digit_chi2_benford": _first_digit_chi2_benford(values),
        "epsilon_benford_deviation": benford_epsilon(values),
        "total_variation_benford": _total_variation_benford(values),
        "benford_probs": {str(d): v for d, v in _benford_probs().items()},
    }

    # ---- repeats + runs (baselined) ----
    def _rep2(vs):
        return _repeated_ngram_count(vs, 2)

    def _rep3(vs):
        return _repeated_ngram_count(vs, 3)

    rep2_base = (
        null_percentile(
            _rep2, values, tail="upper", n_shuffles=n_shuffles,
            statistic_name="repeated_bigram_count", namespace=rng_namespace,
        )
        if include_repeat_baselines else None
    )
    rep3_base = (
        null_percentile(
            _rep3, values, tail="upper", n_shuffles=n_shuffles,
            statistic_name="repeated_trigram_count", namespace=rng_namespace,
        )
        if include_repeat_baselines else None
    )
    mono_base = null_percentile(
        longest_monotone_run, values, tail="upper", n_shuffles=n_shuffles,
        statistic_name="longest_monotone_run", namespace=rng_namespace,
    )
    consec_base = null_percentile(
        consecutive_integer_pairs, values, tail="upper", n_shuffles=n_shuffles,
        statistic_name="consecutive_integer_pairs", namespace=rng_namespace,
    )
    repeats = {
        "repeated_bigrams": _repeated_ngram_positions(values, 2),
        "repeated_trigrams": _repeated_ngram_positions(values, 3),
        "repeated_bigram_count": _rep2(values),
        "repeated_bigram_baseline": rep2_base,
        "repeated_trigram_count": _rep3(values),
        "repeated_trigram_baseline": rep3_base,
    }
    runs = {
        "longest_monotone_run": longest_monotone_run(values),
        "longest_monotone_run_baseline": mono_base,
        "longest_strictly_increasing_run": longest_strictly_increasing_run(values),
        "consecutive_integer_pairs": consecutive_integer_pairs(values),
        "consecutive_integer_pairs_baseline": consec_base,
    }

    # ---- modulo (parametric uniform null) ----
    modulo: dict[str, Any] = {}
    if include_modulo:
        for m in list(range(2, 13)) + [26]:
            def _mod_chi2(vs, mm=m):
                c = Counter(v % mm for v in vs)
                exp = len(vs) / mm
                return sum((c.get(r, 0) - exp) ** 2 / exp for r in range(mm))
            base = parametric_percentile(
                _mod_chi2, values, _uniform_sampler(mn, mx), tail="upper",
                n_shuffles=n_shuffles, statistic_name=f"modulo_m{m}", namespace=rng_namespace,
            )
            modulo[str(m)] = {"chi2": _mod_chi2(values), "baseline": base}

    # ---- key_length + front_loading ----
    fli = front_loading_index(values)
    quint = [0, 0, 0, 0, 0]
    for v in values:
        idx = min(4, int((v / mx) * 5)) if mx > 0 else 0
        quint[idx] += 1
    fl_base = parametric_percentile(
        front_loading_index, values, _uniform_sampler(mn, mx), tail="upper",
        n_shuffles=n_shuffles, statistic_name="front_loading_index", namespace=rng_namespace,
    )
    key_length = {
        "word_position_key_words": mx,
        "char_position_key_chars": mx,
        "skip_nth_feasibility": {str(N): N * mx for N in range(2, 6)},
    }
    front_loading = {
        "front_loading_index": fli,
        "quintile_histogram": quint,
        "baseline": fl_base,
        "significant": (fli > 1.0 and fl_base["p_value"] <= 0.05),
    }

    # ---- flags (split; k-of-n rules with plausibility) ----
    rr = repeat_rate(values)
    rr_null = parametric_percentile(
        repeat_rate, values, _uniform_sampler(mn, mx), tail="lower",
        n_shuffles=n_shuffles, statistic_name="repeat_rate", namespace=rng_namespace,
    )
    repeat_null_mean = rr_null["null_mean"]
    repeat_null_std = rr_null["null_std"]
    repeat_within_1sd = repeat_null_std > 0 and abs(rr - repeat_null_mean) <= repeat_null_std
    below_multinomial = rr < repeat_null_mean

    book_feasible = 1 <= mn and mx <= BOOK_KEYLEN_MAX
    front_present = front_loading["significant"]
    front_or_below = front_present or below_multinomial

    monotone_sig = mono_base["p_value"] <= 0.05
    last_digit_uniform = digits["last_digit_uniform"]
    last_digit_sig = not last_digit_uniform
    consec_sig = consec_base["p_value"] <= 0.05

    def _flag(supported: bool, *, counter: bool = False) -> str:
        if supported:
            return "supported"
        return "counterindicated" if counter else "neutral"

    word_flag = book_feasible and front_or_below
    char_flag = book_feasible and front_or_below
    skip_feasible = (5 * mx) <= (BOOK_KEYLEN_MAX * 5)  # always structurally computable
    skip_flag = book_feasible and skip_feasible and front_or_below

    rand_cond = [
        last_digit_uniform,
        benford["epsilon_benford_deviation"] >= BENFORD_INCONSISTENT_EPSILON,
        repeat_within_1sd,
    ]
    independent_random_like = sum(1 for c in rand_cond if c) >= 2

    hoax_cond = [monotone_sig, last_digit_sig, consec_sig]
    structured_hoax_artifact = sum(1 for c in hoax_cond if c) >= 2

    flags = {
        "word_first_letter_book_cipher": {
            "plausibility": _flag(word_flag),
            "key_length_feasible": book_feasible,
            "front_or_below_multinomial": front_or_below,
        },
        "char_position_book_cipher": {
            "plausibility": _flag(char_flag),
            "key_length_feasible": book_feasible,
            "front_or_below_multinomial": front_or_below,
        },
        "skip_nth_word": {
            "plausibility": _flag(skip_flag),
            "key_length_feasible": book_feasible and skip_feasible,
            "front_or_below_multinomial": front_or_below,
        },
        "independent_random_like": {
            "plausibility": _flag(independent_random_like),
            "conditions_met": sum(1 for c in rand_cond if c),
            "last_digit_uniform": last_digit_uniform,
            "epsilon_benford_high": rand_cond[1],
            "repeat_within_1sd": repeat_within_1sd,
        },
        "structured_hoax_artifact": {
            "plausibility": _flag(structured_hoax_artifact),
            "conditions_met": sum(1 for c in hoax_cond if c),
            "monotone_run_significant": monotone_sig,
            "last_digit_preference_significant": last_digit_sig,
            "consecutive_run_significant": consec_sig,
        },
    }

    # ---- companion ----
    companion: dict[str, Any] | None = None
    if related_profile is not None:
        companion = _related_profile_distance(values, related_profile)
        ref_inv = set(related_profile.get("inventory", []))
        my_inv = set(values)
        jacc = (len(ref_inv & my_inv) / len(ref_inv | my_inv)) if (ref_inv or my_inv) else 0.0
        corr = companion.get("first_digit_correlation", 0.0)
        shared = jacc >= _SHARED_JACCARD_MIN and corr >= _SHARED_CORR_MIN
        flags["shared_key_with_companion"] = {
            "plausibility": _flag(shared),
            "inventory_jaccard": jacc,
            "first_digit_correlation": corr,
        }

    battery = {
        "status": "ok",
        "basic": basic,
        "digits": digits,
        "benford": benford,
        "repeats": repeats,
        "runs": runs,
        "modulo": modulo,
        "key_length": key_length,
        "front_loading": front_loading,
        "flags": flags,
        "repeat_null_mean": repeat_null_mean,
        "repeat_null_std": repeat_null_std,
        "repeat_within_1sd": repeat_within_1sd,
        # flat measurements consumed by the diagnosis atom layer:
        "measurements": {
            "count": n,
            "unique": len(uniq),
            "unique_rate": len(uniq) / n,
            "min": mn,
            "max": mx,
            "repeat_rate": rr,
            "repeat_null_mean": repeat_null_mean,
            "repeat_null_std": repeat_null_std,
            "repeat_within_1sd": repeat_within_1sd,
            "below_multinomial": below_multinomial,
            "front_loading_index": fli,
            "front_loading_significant": front_present,
            "epsilon_benford_deviation": benford["epsilon_benford_deviation"],
            "last_digit_uniform": last_digit_uniform,
            "monotone_run_significant": monotone_sig,
            "consecutive_run_significant": consec_sig,
            "book_keylength_plausible": book_feasible,
            "independent_random_like": independent_random_like,
            "structured_hoax_artifact": structured_hoax_artifact,
            "word_first_letter_book_cipher": word_flag,
            "char_position_book_cipher": char_flag,
            "skip_nth_word": skip_flag,
        },
    }
    if companion is not None:
        battery["related_profile_distance"] = companion
    return battery


def _related_profile_distance(values: Sequence[int], ref: dict[str, Any]) -> dict[str, Any]:
    """Distance of this stream's basic/digit profile from a reference cipher's."""
    n = len(values)
    my_first = [Counter(_first_digit(v) for v in values).get(d, 0) / n for d in range(1, 10)]
    ref_first = ref.get("first_digit_dist_norm")
    corr = 0.0
    if ref_first and len(ref_first) == 9:
        mm = sum(my_first) / 9
        rm = sum(ref_first) / 9
        num = sum((a - mm) * (b - rm) for a, b in zip(my_first, ref_first))
        da = math.sqrt(sum((a - mm) ** 2 for a in my_first))
        db = math.sqrt(sum((b - rm) ** 2 for b in ref_first))
        corr = num / (da * db) if da > 0 and db > 0 else 0.0
    return {
        "benford_epsilon_self": benford_epsilon(values),
        "benford_epsilon_ref": ref.get("epsilon_benford_deviation"),
        "max_self": max(values),
        "max_ref": ref.get("max"),
        "first_digit_correlation": corr,
    }


def profile_for_related(values: Sequence[int]) -> dict[str, Any]:
    """Build the reference profile another cipher can pass as ``related_profile``.

    Carries only this cipher's OWN measurements — never any plaintext/key text.
    """
    values = list(values)
    n = len(values)
    return {
        "inventory": list(set(values)),
        "max": max(values) if values else 0,
        "min": min(values) if values else 0,
        "epsilon_benford_deviation": benford_epsilon(values),
        "first_digit_dist_norm": [
            Counter(_first_digit(v) for v in values).get(d, 0) / n for d in range(1, 10)
        ] if n else [0.0] * 9,
    }


# ---------------------------------------------------------------------------
# Gillogly alphabetical-run detector (Part 4)
# ---------------------------------------------------------------------------

def _longest_alpha_run(letters: str, *, strict: bool) -> tuple[int, int]:
    """(length, start_offset) of the longest (non-)decreasing alphabetical run."""
    if not letters:
        return 0, 0
    best_len = cur = 1
    best_start = start = 0
    for i in range(1, len(letters)):
        step_ok = letters[i] > letters[i - 1] if strict else letters[i] >= letters[i - 1]
        if step_ok:
            cur += 1
            if cur > best_len:
                best_len = cur
                best_start = start
        else:
            cur = 1
            start = i
    return best_len, best_start


def alphabetical_run_report(
    letters: str,
    *,
    n_shuffles: int = 1000,
    rng_namespace: str = "",
) -> dict[str, Any]:
    """Gillogly-1980 fabrication detector on a decoded first-letter stream.

    Reports the longest non-decreasing and strictly-increasing alphabetical runs,
    their offsets, and baselined:upper percentiles at ``n_shuffles`` (default 1000)
    against a frequency-preserving letter shuffle. A long ascending run at
    ``p < 1/(n_shuffles+1)`` is the signature of a fabricated (non-plaintext)
    stream.
    """
    upper = "".join(ch for ch in letters.upper() if "A" <= ch <= "Z")
    if not upper:
        return {"status": "not_computable", "reason": "no_letters", "length": 0}

    codes = [ord(c) - 65 for c in upper]

    nondec_len, nondec_off = _longest_alpha_run(upper, strict=False)
    incr_len, incr_off = _longest_alpha_run(upper, strict=True)

    def _nondec(vs):
        best = cur = 1
        for i in range(1, len(vs)):
            if vs[i] >= vs[i - 1]:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best

    def _incr(vs):
        best = cur = 1
        for i in range(1, len(vs)):
            if vs[i] > vs[i - 1]:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best

    nondec_base = null_percentile(
        _nondec, codes, tail="upper", n_shuffles=n_shuffles,
        statistic_name="alpha_nondecreasing_run", namespace=rng_namespace,
    )
    incr_base = null_percentile(
        _incr, codes, tail="upper", n_shuffles=n_shuffles,
        statistic_name="alpha_increasing_run", namespace=rng_namespace,
    )
    return {
        "status": "ok",
        "length": len(upper),
        "longest_nondecreasing_run": nondec_len,
        "longest_nondecreasing_offset": nondec_off,
        "longest_nondecreasing_text": upper[nondec_off : nondec_off + nondec_len],
        "longest_nondecreasing_baseline": nondec_base,
        "longest_increasing_run": incr_len,
        "longest_increasing_offset": incr_off,
        "longest_increasing_text": upper[incr_off : incr_off + incr_len],
        "longest_increasing_baseline": incr_base,
    }
