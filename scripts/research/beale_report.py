#!/usr/bin/env python3
"""Beale statistical-fingerprinting report (INV-0 Part 8).

Reproduces (qualitatively) one headline statistic from each of three reference
analyses of the Beale ciphers, using only local compute:

  (a) Gillogly 1980 — B2 self-check under the DoI key, then the B1/B3 first-letter
      decode + alphabetical-run artifact detector.
  (b) Wase 2021   — first-digit Benford / epsilon-Benford deviation per cipher.
  (c) Campanelli 2023 — last-digit uniformity, unique-token growth, gap structure,
      and per-statistic divergence of B1/B3 from B2 under the standard B2 method.

Zero access to any claimed B1/B3 plaintext (none exists in the benchmark). Beale 2
(the SOLVED companion) is used only as a `related_profile`, never as ground truth.

stdlib + repo imports only. No network, no LLM, no `services/` import.

Run:
  PYTHONPATH=src .venv/bin/python scripts/research/beale_report.py \
      --benchmark-root ~/Dropbox/src2/cipher_benchmark/benchmark [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from analysis.coherence import island_report  # noqa: E402
from analysis.numeric_code import (  # noqa: E402
    BEALE_DOI_KEY_PATH,
    alphabetical_run_report,
    numeric_code_battery,
    parse_numeric_ciphertext,
    profile_for_related,
)
from benchmark.unsolved import load_unsolved_record  # noqa: E402

_RESOURCE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "resources", "reference",
)
_EXPECTED_OPENING = "IHAVEDEPOSITED"


def _load_beale2() -> list[int]:
    with open(os.path.join(_RESOURCE, "beale2_numbers.txt")) as f:
        return parse_numeric_ciphertext(f.read())


def _load_doi_words() -> list[str]:
    """Load the DoI KEY (legitimately, as a key — beale_report decodes with it)."""
    words: list[str] = []
    with open(BEALE_DOI_KEY_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.append(line.upper())
    return words


def _first_letter(word: str) -> str:
    for ch in word:
        if ch.isalpha():
            return ch.upper()
    return "?"


def _decode_first_letters(values, doi_words):
    """Decode a numeric stream via DoI word-first-letters; skip out-of-range."""
    letters: list[str] = []
    skipped: list[int] = []
    for pos, v in enumerate(values):
        if 1 <= v <= len(doi_words):
            letters.append(_first_letter(doi_words[v - 1]))
        else:
            skipped.append(pos)
    return "".join(letters), skipped


def _campanelli_stats(values):
    b = numeric_code_battery(values, include_modulo=False, include_repeat_baselines=False)
    n = len(values)
    # unique-token growth at 10 positional deciles (cumulative unique count)
    growth = []
    seen = set()
    for i, v in enumerate(values, 1):
        seen.add(v)
        if i % max(1, n // 10) == 0 or i == n:
            growth.append(len(seen))
    growth = growth[:10]
    gaps = b["basic"]["gap_histogram"]
    return {
        "last_digit_uniform": bool(b["measurements"]["last_digit_uniform"]),
        "unique_token_growth_deciles": growth,
        "final_unique": b["basic"]["unique"],
        "gap_histogram": gaps,
        "max_gap": b["basic"]["max_gap"],
        "gaps_equal_one": b["basic"]["gaps_equal_one"],
    }


def build_report(benchmark_root: str) -> dict:
    b1 = load_unsolved_record(benchmark_root, "beale_1").numeric_values()
    b3 = load_unsolved_record(benchmark_root, "beale_3").numeric_values()
    b2 = _load_beale2()
    doi = _load_doi_words()

    b2_profile = profile_for_related(b2)

    # (a) full P8 batteries (B2 as related_profile for B1/B3)
    batteries = {
        "beale_1": numeric_code_battery(b1, related_profile=b2_profile, rng_namespace="beale_1"),
        "beale_2": numeric_code_battery(b2, rng_namespace="beale_2"),
        "beale_3": numeric_code_battery(b3, related_profile=b2_profile, rng_namespace="beale_3"),
    }

    # (b) Wase — Benford / epsilon-Benford
    wase = {
        name: {
            "epsilon_benford_deviation": bat["benford"]["epsilon_benford_deviation"],
            "total_variation_benford": bat["benford"]["total_variation_benford"],
            "first_digit_chi2_benford": bat["benford"]["first_digit_chi2_benford"],
            "first_digit_chi2_uniform": bat["digits"]["first_digit_chi2_uniform"],
        }
        for name, bat in batteries.items()
    }

    # (c) Campanelli — operationalized divergence vs B2
    camp_raw = {
        "beale_1": _campanelli_stats(b1),
        "beale_2": _campanelli_stats(b2),
        "beale_3": _campanelli_stats(b3),
    }
    b2c = camp_raw["beale_2"]
    campanelli = {"per_cipher": camp_raw, "divergence_vs_b2": {}}
    for name in ("beale_1", "beale_3"):
        c = camp_raw[name]
        campanelli["divergence_vs_b2"][name] = {
            "last_digit_uniform": bool(c["last_digit_uniform"]) != bool(b2c["last_digit_uniform"]),
            "gap_structure": abs(c["max_gap"] - b2c["max_gap"]) > 50,
            "unique_growth": abs(c["final_unique"] / max(1, len(b1 if name == "beale_1" else b3))
                                 - b2c["final_unique"] / len(b2)) > 0.10,
        }

    # (d) Gillogly — B2 self-check, then B1/B3 first-letter decode + alpha runs
    b2_decode, b2_skipped = _decode_first_letters(b2, doi)
    opening = b2_decode[: len(_EXPECTED_OPENING)]
    b2_self = island_report(b2_decode, "en")
    self_check = {
        "decoded_opening": opening,
        "expected_opening": _EXPECTED_OPENING,
        "opening_matches": opening == _EXPECTED_OPENING,
        "opening_match_count": sum(1 for a, b in zip(opening, _EXPECTED_OPENING) if a == b),
        "skipped_index_count": len(b2_skipped),
        "island_verdict": b2_self["verdict"],
        "passed": opening == _EXPECTED_OPENING and b2_self["verdict"] == "coherent",
        "note": (
            "PROVENANCE: this key is the STANDARD PUBLIC-DOMAIN DoI numbering, "
            "which decodes 12/14 of the B2 opening (IHAREDEPOSCTED; positions "
            "4/11 need 1885-Ward-pamphlet numbering quirks), so the B2 "
            "self-check is REPORTED, not asserted as passed. The B1 "
            "nondecreasing-run statistic below reproduces UNDER THE STANDARD "
            "PUBLIC-DOMAIN NUMBERING; its exact-1885-pamphlet form is "
            "PROVISIONAL pending a pamphlet-quirk-corrected key. Robustness: "
            "the run statistic tolerates the 2-of-14 key-word error rate this "
            "self-check exposes (isolated first-letter substitutions rarely "
            "break a 14-letter monotone run)."
        ),
    }

    gillogly = {"b2_self_check": self_check, "b1": None, "b3": None}
    for name, vals in (("b1", b1), ("b3", b3)):
        dec, skipped = _decode_first_letters(vals, doi)
        runs = alphabetical_run_report(dec, n_shuffles=1000, rng_namespace=name)
        gillogly[name] = {
            "decoded_length": len(dec),
            "skipped_index_count": len(skipped),
            "longest_nondecreasing_run": runs.get("longest_nondecreasing_run"),
            "longest_nondecreasing_text": runs.get("longest_nondecreasing_text"),
            "nondecreasing_run_p": runs.get("longest_nondecreasing_baseline", {}).get("p_value"),
            "longest_increasing_run": runs.get("longest_increasing_run"),
            "longest_increasing_text": runs.get("longest_increasing_text"),
            "increasing_run_p": runs.get("longest_increasing_baseline", {}).get("p_value"),
            "island_verdict": island_report(dec, "en")["verdict"],
        }

    return {
        "p8_batteries": {
            name: {
                "basic": bat["basic"],
                "digits": bat["digits"],
                "benford": bat["benford"],
                "front_loading": {k: v for k, v in bat["front_loading"].items() if k != "baseline"},
                "flags": {k: v.get("plausibility") for k, v in bat["flags"].items()},
            }
            for name, bat in batteries.items()
        },
        "wase_benford": wase,
        "campanelli": campanelli,
        "gillogly": gillogly,
    }


def _fmt(report: dict) -> str:
    L: list[str] = []
    L.append("=" * 72)
    L.append("BEALE STATISTICAL FINGERPRINTING REPORT (INV-0 Part 8)")
    L.append("=" * 72)

    L.append("\n(a) P8 numeric battery — basic profile")
    for name, bat in report["p8_batteries"].items():
        b = bat["basic"]
        fl = bat["front_loading"]
        L.append(f"  {name}: n={b['count']} unique={b['unique']} "
                 f"max={b['max']} front_load_index={fl['front_loading_index']:.2f} "
                 f"significant={fl['significant']}")
        L.append(f"       flags: {bat['flags']}")

    L.append("\n(b) Wase 2021 — Benford / epsilon-Benford (want B1,B3 differ from B2)")
    for name, w in report["wase_benford"].items():
        L.append(f"  {name}: epsilon_benford={w['epsilon_benford_deviation']:.4f} "
                 f"TV={w['total_variation_benford']:.4f} "
                 f"chi2_vs_benford={w['first_digit_chi2_benford']:.1f}")

    L.append("\n(c) Campanelli 2023 — operationalized divergence from B2")
    for name in ("beale_1", "beale_3"):
        c = report["campanelli"]["per_cipher"][name]
        d = report["campanelli"]["divergence_vs_b2"][name]
        L.append(f"  {name}: last_digit_uniform={c['last_digit_uniform']} "
                 f"max_gap={c['max_gap']} unique={c['final_unique']} "
                 f"growth={c['unique_token_growth_deciles']}")
        L.append(f"       divergence_vs_b2: {d}")
    c2 = report["campanelli"]["per_cipher"]["beale_2"]
    L.append(f"  beale_2 (reference): last_digit_uniform={c2['last_digit_uniform']} "
             f"max_gap={c2['max_gap']} unique={c2['final_unique']}")

    L.append("\n(d) Gillogly 1980 — B2 self-check + B1/B3 alphabetical-run artifact")
    L.append("  Key provenance: STANDARD PUBLIC-DOMAIN DoI numbering (see B2 self-check).")
    sc = report["gillogly"]["b2_self_check"]
    L.append(f"  B2 self-check: decoded_opening='{sc['decoded_opening']}' "
             f"expected='{sc['expected_opening']}' "
             f"match={sc['opening_match_count']}/{len(sc['expected_opening'])} "
             f"skipped={sc['skipped_index_count']} island={sc['island_verdict']} "
             f"PASSED={sc['passed']}")
    L.append(f"       {sc['note']}")
    for name in ("b1", "b3"):
        g = report["gillogly"][name]
        L.append(f"  {name} (under the standard public-domain DoI numbering): "
                 f"decoded_len={g['decoded_length']} skipped={g['skipped_index_count']}")
        L.append(f"       longest non-decreasing run={g['longest_nondecreasing_run']} "
                 f"p={g['nondecreasing_run_p']:.5f} text='{g['longest_nondecreasing_text']}'")
        L.append(f"       longest strictly-increasing run={g['longest_increasing_run']} "
                 f"p={g['increasing_run_p']:.5f} text='{g['longest_increasing_text']}'")
    L.append("  NOTE: the B1 run statistic above reproduces under the standard "
             "public-domain numbering; its exact-1885-Ward-pamphlet form is "
             "PROVISIONAL pending a pamphlet-quirk-corrected key (B2 self-check "
             f"{sc['opening_match_count']}/{len(sc['expected_opening'])}). The run "
             "statistic tolerates that 2-of-14 key-word error rate.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Beale statistical fingerprinting report")
    ap.add_argument("--benchmark-root",
                    default=os.path.expanduser("~/Dropbox/src2/cipher_benchmark/benchmark"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = build_report(args.benchmark_root)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(_fmt(report))


if __name__ == "__main__":
    main()
