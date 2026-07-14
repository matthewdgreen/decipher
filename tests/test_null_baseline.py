"""Shuffle-null baseline tests (INV-0 Part 3 / Part 9)."""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.null_baseline import (
    derive_seed,
    encode_tokens,
    null_percentile,
    parametric_percentile,
    tokens_digest,
)


def _nondecreasing_run(values):
    best = cur = 1
    for i in range(1, len(values)):
        if values[i] >= values[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best if values else 0


def _adjacent_equal(values):
    return sum(1 for i in range(len(values) - 1) if values[i] == values[i + 1])


def test_encode_tokens_handles_large_values():
    # Beale's max value 2906 overflows a byte; 8-byte-BE must round-trip.
    enc = encode_tokens([0, 255, 256, 2906])
    assert len(enc) == 8 * 4
    assert enc == b"".join(v.to_bytes(8, "big") for v in [0, 255, 256, 2906])


def test_deterministic_seed_and_digest():
    toks = [3, 1, 4, 1, 5, 9, 2906, 6]
    assert derive_seed("s", "", toks) == derive_seed("s", "", toks)
    assert tokens_digest(toks) == tokens_digest(list(toks))
    # namespace changes the seed
    assert derive_seed("s", "a", toks) != derive_seed("s", "b", toks)


def test_determinism_across_processes():
    toks = [7, 2, 9, 2906, 4, 1, 8, 3, 6, 5]
    local = null_percentile(_nondecreasing_run, toks, tail="upper", n_shuffles=200,
                            statistic_name="run")
    code = (
        "import os, sys\n"
        "sys.path.insert(0, os.path.join(os.getcwd(), 'src'))\n"
        "from analysis.null_baseline import null_percentile\n"
        "def f(v):\n"
        "    b = c = 1\n"
        "    for i in range(1, len(v)):\n"
        "        if v[i] >= v[i-1]:\n"
        "            c += 1; b = max(b, c)\n"
        "        else:\n"
        "            c = 1\n"
        "    return b\n"
        f"r = null_percentile(f, {toks}, tail='upper', n_shuffles=200, statistic_name='run')\n"
        "print(r['seed'], round(r['p_value'], 6), round(r['null_mean'], 6))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    assert out.returncode == 0, out.stderr
    seed, p, mean = out.stdout.split()
    # Seed is exact across processes; p/mean printed rounded to 6 decimals.
    assert int(seed) == local["seed"]
    assert abs(float(p) - local["p_value"]) < 1e-6
    assert abs(float(mean) - local["null_mean"]) < 1e-6


def test_real_upper_tail_signal_significant():
    ordered = list(range(200))  # perfectly non-decreasing -> extreme upper statistic
    res = null_percentile(_nondecreasing_run, ordered, tail="upper", n_shuffles=1000,
                          statistic_name="run_up")
    assert res["p_value"] <= 0.05
    assert res["observed"] == 200


def test_significantly_low_statistic_lower_tail():
    # An alternating stream has zero adjacent-equal pairs — significantly LOW vs
    # a frequency-preserving shuffle of the same multiset (finding 6a).
    alt = [0, 1] * 100
    res = null_percentile(_adjacent_equal, alt, tail="lower", n_shuffles=1000,
                          statistic_name="adj_eq")
    assert res["observed"] == 0
    assert res["p_value"] <= 0.05


def test_n_shuffles_1000_supports_999_bound():
    ordered = list(range(300))
    res = null_percentile(_nondecreasing_run, ordered, tail="upper", n_shuffles=1000,
                          statistic_name="bound")
    # When the observation exceeds every draw the tightest bound is 1/(n+1).
    assert res["n_shuffles"] == 1000
    assert res["p_value"] <= 1.0 / 1001 + 1e-12


def test_parametric_null_labeled():
    def stat(v):
        return sum(v)
    res = parametric_percentile(
        stat, [1, 2, 3, 4, 5], lambda rng, n: [rng.randint(1, 5) for _ in range(n)],
        tail="upper", n_shuffles=100, statistic_name="sum_par",
    )
    assert res["kind"] == "parametric"
    assert 0.0 < res["p_value"] <= 1.0
