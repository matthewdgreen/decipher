"""Opt-in R5 periodic probes; ciphertext-only rules frozen before evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

from analysis.cipher_id import compute_cipher_fingerprint, estimate_fundamental_period
from automated.bounded_process import run_guarded

VARIANTS = ("vigenere", "beaufort", "variant_beaufort")
WALL_SECONDS = 45
MAX_CANDIDATES = 6
SEED = 61001


def content_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def routing_mode():
    mode = os.environ.get("DECIPHER_PERIODIC_ROUTING", "off")
    if mode not in {"off", "probe_v1"}:
        raise ValueError("DECIPHER_PERIODIC_ROUTING must be off or probe_v1")
    return mode


def select_periods(table, global_ic, token_count, kasiski):
    table = {int(p): float(ic) for p, ic in table.items()
             if 2 <= int(p) <= min(20, token_count // 25) and math.isfinite(ic)}
    if not table:
        return {"periodic_ic": {}, "periods": [], "reason": "no_reliable_phases"}
    null = statistics.median(table.values())
    fundamental, harmonic = estimate_fundamental_period(
        table, token_count, max_period=max(table), null_ic=null, kasiski_factors=kasiski)
    qualifying = [p for p, ic in table.items()
                  if ic >= .055 and ic - global_ic >= .010 and ic - null >= .010]
    qualifying.sort(key=lambda p: (-(table[p] - null), p))
    if fundamental in qualifying:
        qualifying.remove(fundamental)
        qualifying.insert(0, fundamental)
    return {"periodic_ic": table, "global_ic": global_ic, "median_ic": null,
            "fundamental": fundamental, "harmonic": harmonic, "kasiski": kasiski,
            "periods": qualifying[:2],
            "reason": "periodic_evidence" if qualifying else "no_qualifying_period"}


def diagnose(cipher, language, *, cipher_system="", solver_hints=None,
             transform_pipeline=None, transform_search="off", model_variant=None):
    reason = None
    symbols = [cipher.alphabet.symbol_for(t) for t in cipher.tokens]
    custom = sorted(k for k in os.environ if k.startswith((
        "DECIPHER_KEYED_VIGENERE_", "DECIPHER_POLYALPHABETIC_", "DECIPHER_QUAGMIRE_"))
        and not (k == "DECIPHER_QUAGMIRE_THREADS" and os.environ[k] == "4")
        and not (k == "DECIPHER_QUAGMIRE_SEARCH_SEED" and os.environ[k] == str(SEED)))
    if language != "en":
        reason = "language_out_of_scope"
    elif cipher_system.strip() or solver_hints or transform_pipeline is not None or transform_search != "off":
        reason = "explicit_direction_preserved"
    elif custom or model_variant is not None:
        reason = "custom_search_configuration_preserved"
    elif not 180 <= len(symbols) <= 1200:
        reason = "length_out_of_scope"
    elif any(len(s) != 1 or not "A" <= s <= "Z" for s in symbols):
        reason = "non_a_z_symbols"
    base = {"eligible": reason is None, "token_count": len(symbols),
            "periods": [], "reason": reason, "custom_search_keys": custom}
    if reason is None:
        fingerprint = compute_cipher_fingerprint(cipher.tokens, cipher.alphabet.size,
            max_period=min(20, len(symbols) // 25), language=language,
            word_group_count=len(cipher.words))
        base.update(select_periods(fingerprint.periodic_ic, fingerprint.ic,
                                   len(symbols), fingerprint.kasiski_spacing_gcds))
    return base


def replay_matches(ciphertext, candidate):
    """Strict correspondence, never a claim of cipher truth or readability."""
    from analysis.polyalphabetic import decode_values, encode_quagmire_plaintext
    text, period = candidate.get("plaintext"), candidate.get("period")
    if (not isinstance(text, str) or len(text) != len(ciphertext)
            or any(not "A" <= c <= "Z" for c in text)
            or type(period) is not int or not 2 <= period <= 20):
        return False
    try:
        if candidate.get("engine") == "periodic":
            shifts = candidate.get("shifts")
            variant = candidate.get("variant")
            if (variant not in VARIANTS or not isinstance(shifts, list) or len(shifts) != period
                    or any(type(s) is not int or not 0 <= s < 26 for s in shifts)):
                return False
            if candidate.get("key") != "".join(chr(65 + s) for s in shifts):
                return False
            return decode_values([ord(c) - 65 for c in ciphertext], shifts, variant=variant) == text
        if candidate.get("engine") == "quagmire3":
            meta = candidate.get("metadata") or {}
            keyword, cycleword = meta.get("alphabet_keyword"), meta.get("cycleword")
            if (not isinstance(keyword, str) or len(keyword) not in (6, 7, 8)
                    or len(set(keyword)) != len(keyword)
                    or not isinstance(cycleword, str) or len(cycleword) != period
                    or any(not "A" <= c <= "Z" for c in keyword + cycleword)):
                return False
            alphabet = "".join(dict.fromkeys(keyword + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            shifts = [alphabet.index(c) for c in cycleword]
            if (candidate.get("key") != cycleword or candidate.get("shifts") != shifts
                    or any(meta.get(k, alphabet) != alphabet for k in ("plaintext_alphabet", "ciphertext_alphabet"))
                    or meta.get("cycleword_shifts", shifts) != shifts):
                return False
            return encode_quagmire_plaintext(text, cycleword=cycleword,
                                             alphabet_keyword=keyword) == ciphertext
    except (ValueError, TypeError, KeyError, AttributeError):
        return False
    return False


def language_gate(validation):
    def numeric(value):
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
    seg = validation.get("segmentation") or {}
    values = [validation.get("validation_score"), validation.get("quadgram_loglik_per_gram"),
              validation.get("strict_word_hit_score"), seg.get("dict_rate"), seg.get("pseudo_word_fraction")]
    return (all(numeric(v) for v in values) and validation.get("validation_label") == "coherent_candidate"
            and values[2] >= .25 and values[3] >= .78 and values[4] <= .25)


def rank_key(row):
    validation = row.get("validation") or {}
    def number(key):
        v = validation.get(key)
        return v if isinstance(v, (int, float)) and math.isfinite(v) else -math.inf
    period = row.get("period")
    return (not row.get("delivery_eligible", False), -number("validation_score"),
            -number("quadgram_loglik_per_gram"), period if type(period) is int else 99, row["content_hash"])


def retain(rows):
    result, seen = [], set()
    for row in sorted(rows, key=rank_key):
        if row["content_hash"] not in seen:
            result.append(row)
            seen.add(row["content_hash"])
        if len(result) == MAX_CANDIDATES:
            break
    return result


def probe_search(request, emit, *, periodic=None, quagmire=None, validator=None):
    """Worker-side search. Each published menu is complete evidence on timeout."""
    from analysis import polyalphabetic
    from analysis.finalist_validation import validate_plaintext_finalist
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText
    if set(request) != {"ciphertext", "periods", "language"}:
        raise ValueError("probe request violates runtime allowlist")
    text, periods = request["ciphertext"], request["periods"]
    if (request["language"] != "en" or not isinstance(text, str) or not 180 <= len(text) <= 1200
            or any(not "A" <= c <= "Z" for c in text) or not isinstance(periods, list)
            or not 1 <= len(periods) <= 2 or len(set(periods)) != len(periods)
            or any(type(p) is not int or not 2 <= p <= min(20, len(text) // 25) for p in periods)):
        raise ValueError("invalid bounded probe input")
    cipher = CipherText(raw=text, alphabet=Alphabet.from_text(text), separator=None)
    periodic = periodic or polyalphabetic.search_periodic_polyalphabetic
    validator = validator or validate_plaintext_finalist
    menu = []
    for engine in ("periodic", "quagmire3"):
        emit({"event": "stage", "engine": engine})
        if engine == "periodic":
            result = periodic(cipher, language="en", periods=periods, variants=list(VARIANTS),
                              top_n=MAX_CANDIDATES, refine=True)
        else:
            if quagmire is None:
                from analysis.polyalphabetic_fast import search_quagmire3_shotgun_fast
                quagmire = search_quagmire3_shotgun_fast
            result = quagmire(cipher, language="en", keyword_lengths=[6, 7, 8],
                cycleword_lengths=periods, hillclimbs=5000, restarts=250, threads=4,
                seed=SEED, slip_probability=.001, backtrack_probability=.15,
                initial_keywords=[], top_n=MAX_CANDIDATES)
        for raw in (result.get("top_candidates") or [])[:MAX_CANDIDATES]:
            if not isinstance(raw, dict) or not isinstance(raw.get("plaintext"), str):
                continue
            # Preserve mode-specific state, but not arbitrary nested result payloads.
            row = {k: raw[k] for k in ("plaintext", "period", "variant", "shifts", "key", "metadata") if k in raw}
            row.update(engine=engine, content_hash=content_hash(row["plaintext"]),
                       solver=result.get("solver"))
            row["replay_consistent"] = row.get("period") in periods and replay_matches(text, row)
            row["validation"] = validator(row["plaintext"], language="en") if row["replay_consistent"] else {}
            row["delivery_eligible"] = row["replay_consistent"] and language_gate(row["validation"])
            menu = retain([*menu, row])
            emit({"event": "menu", "candidates": menu})
        if any(r["delivery_eligible"] for r in menu):
            break
    emit({"event": "finished", "candidates": menu})


def probe_environment():
    # Keep only local runtime resource resolution, never provider credentials or
    # custom search overrides. Arguments freeze all native search knobs.
    env = {k: v for k, v in os.environ.items()
           if k in {"PATH", "TMPDIR", "SYSTEMROOT", "DECIPHER_BOUNDED_REGISTRY", "DECIPHER_BOUNDED_ANCESTORS"}
           or k.startswith("DECIPHER_NGRAM_MODEL_")}
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONNOUSERSITE="1",
               PYTHONUNBUFFERED="1", PYTHONHASHSEED=str(SEED), LC_ALL="C", LANG="C")
    for key in ("DECIPHER_PARALLEL_WORKERS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "4"
    return env


def run_probe(cipher, diagnosis):
    ciphertext = "".join(cipher.alphabet.symbol_for(t) for t in cipher.tokens)
    request = {"ciphertext": ciphertext, "language": "en", "periods": diagnosis["periods"]}
    with tempfile.TemporaryDirectory(prefix="decipher-periodic-") as directory:
        execution = run_guarded([sys.executable, "-m", "automated.periodic_probe"], request,
            env=probe_environment(), cwd=directory, wall_seconds=WALL_SECONDS)
    menu = []
    for event in execution.get("events", []):
        if event.get("event") in {"menu", "finished"}:
            for row in event.get("candidates", []):
                if not isinstance(row, dict) or not isinstance(row.get("plaintext"), str):
                    continue
                row = dict(row)
                row["content_hash"] = content_hash(row["plaintext"])
                row["replay_consistent"] = row.get("period") in diagnosis["periods"] and replay_matches(ciphertext, row)
                row["delivery_eligible"] = row["replay_consistent"] and language_gate(row.get("validation") or {})
                menu = retain([*menu, row])
    events = execution.get("events", [])
    finished = [e for e in events if e.get("event") == "finished"]
    if execution["execution_status"] == "completed" and (len(finished) != 1 or events[-1] != finished[0]):
        execution["execution_status"] = "worker_error"
    selected = (next((r for r in menu if r["delivery_eligible"]), None)
                if execution["execution_status"] == "completed" and execution["cleanup_confirmed"] else None)
    return {"name": "probe_periodic_routing", "mode": "probe_v1", "diagnosis": diagnosis,
            "execution": {k: v for k, v in execution.items() if k != "events"},
            "stages": [e["engine"] for e in execution.get("events", []) if e.get("event") == "stage"],
            "wall_limit_seconds": WALL_SECONDS, "seed": SEED, "threads": 4,
            "candidates": menu, "selected": selected,
            "decision": "adopt_unverified_candidate" if selected else "fallback",
            "verification": "not_run; replay/language validation is not solved acceptance"}


def selected_step(row):
    """Use existing mode-specific artifact keys for downstream branch import."""
    meta = row.get("metadata") or {}
    quagmire = row["engine"] == "quagmire3"
    return {"name": "search_quagmire3_keyword_alphabet" if quagmire else "search_periodic_polyalphabetic",
            "solver": row["solver"], "status": "completed", "routing": "blind_periodic_probe_v1",
            "period": row["period"], "variant": row.get("variant"),
            "key_type": "QuagmireKey" if quagmire else "PeriodicShiftKey",
            "key": row.get("key"), "shifts": row.get("shifts"),
            "cycleword": meta.get("cycleword"), "alphabet_keyword": meta.get("alphabet_keyword"),
            "plaintext_alphabet": meta.get("plaintext_alphabet"),
            "ciphertext_alphabet": meta.get("ciphertext_alphabet"),
            "quagmire_type": "quag3" if quagmire else None,
            "top_candidates": [row], "seed": SEED,
            "note": "Unverified periodic candidate; no substitution mapping or solution declaration."}


if __name__ == "__main__":
    import resource
    from contextlib import redirect_stdout
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_CANDIDATES * 1024 * 1024,) * 2)
    output = sys.stdout
    with redirect_stdout(sys.stderr):
        probe_search(json.load(sys.stdin), lambda event: print(json.dumps(event), file=output, flush=True))
