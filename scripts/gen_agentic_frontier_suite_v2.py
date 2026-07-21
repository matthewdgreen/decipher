"""Generate the agentic frontier suite v2: 18 fresh contamination-free cases.

Extends v1 (gen_agentic_frontier_suite.py, 9 cases) per the 2026-07-20 coverage
review: replication seeds for the flagship families, post-fix expectation flips
(keyed columnar, Vigenère), and new probe classes — padding-tail declaration,
short-text restraint, misleading-label anchoring, German/French language
probes, a geometric-layer composite, and one historical Borg page.

All v1 plaintexts are BURNED (full decodes appear in the graded results doc),
so every synthetic case here uses fresh original prose and fresh keys.

Outputs:
  docs/evidence/agentic_frontier_suite_v2.md          (pasteable, no answers)
  ~/.config/decipher/dogfood_answers/agentic_frontier_answers_v2.json (sealed)

`--validate` additionally runs the $0 local checks (automated solver / shotgun)
and prints a measured-behavior table so expectations are evidence-based.
"""
import argparse
import json
import random
import string
import sys
from pathlib import Path

REPO = Path.home() / "Dropbox/src2/decipher"
sys.path.insert(0, str(REPO / "src"))

from analysis.polyalphabetic import encode_plaintext, encode_quagmire_plaintext, parse_periodic_key
from ciphers.transposition import RailfenceCipher, columnar_encrypt
from ciphers.fractionation import BifidCipher
from analysis.transformers import TransformPipeline, make_inverse_input_for_pipeline

rng = random.Random(0xBEEF)


def rand_perm(alpha=string.ascii_uppercase):
    p = list(alpha)
    rng.shuffle(p)
    return dict(zip(alpha, p))


def rand_kw(n):
    letters = list(string.ascii_uppercase)
    rng.shuffle(letters)
    return "".join(letters[:n])


# ---- fresh original prose (contamination-free; no v1/probe topic reuse) ----
PROSE = {
 "warmup_mono": "THE LAST LAMPLIGHTER IN THE VALLEY REFUSED TO RETIRE UNTIL EVERY STREET HAD ELECTRIC LIGHT AND THEN HE WALKED HIS ROUTE ONE FINAL EVENING GREETING EACH DARK POLE LIKE AN OLD FRIEND",
 "composite_a": "THE GLASSBLOWER NEVER SOLD THE FIRST PIECE FROM A NEW BATCH OF SAND BECAUSE SHE SAID EVERY SHORE KEEPS A DIFFERENT SILENCE AND THE FIRST VESSEL REMEMBERS IT TOO CLEARLY TO BELONG TO ANYONE ELSE SO HER WORKSHOP SHELVES HELD A ROW OF UNSOLD BOWLS EACH RINGING FAINTLY WITH A BEACH NOBODY COULD NAME AND COLLECTORS OFFERED FORTUNES FOR THE ROW AND WERE ALWAYS REFUSED AND WHEN SHE FINALLY CLOSED THE WORKSHOP SHE CARRIED EVERY BOWL BACK TO ITS OWN COAST AND LEFT EACH ONE ON THE TIDE LINE WHERE ITS SAND HAD BEEN GATHERED",
 "composite_b": "THE ORCHARD KEEPER GRAFTED ONE BRANCH OF EVERY LOST VARIETY ONTO A SINGLE ANCIENT TREE UNTIL IT CARRIED FORTY KINDS OF APPLE AND BLOOMED FOR A FULL MONTH EACH SPRING AND PEOPLE CAME FROM THREE COUNTIES JUST TO STAND UNDER IT AND ARGUE ABOUT WHICH BLOSSOM SMELLED OLDEST",
 "quagmire_a": "THE VIOLIN MAKER KEPT A DRAWER OF WOOD TOO BEAUTIFUL TO USE AND EVERY WINTER HE OPENED IT AND TOUCHED EACH PIECE AND CLOSED IT AGAIN AND HIS APPRENTICE FINALLY UNDERSTOOD THAT THE DRAWER WAS NOT A STORE OF MATERIAL BUT A MUSEUM OF INSTRUMENTS THAT WOULD ALWAYS BE PERFECT BECAUSE THEY WOULD NEVER BE BUILT AND WHEN THE OLD MAN DIED THE APPRENTICE BURNED NOTHING AND BUILT NOTHING FROM THAT DRAWER AND SIMPLY ADDED ONE PIECE OF HIS OWN EACH YEAR AS THE TRADITION QUIETLY REQUIRED",
 "quagmire_b": "THE BRIDGE ENGINEER WALKED ACROSS EVERYTHING SHE BUILT ON THE MORNING IT OPENED CARRYING HER GRANDMOTHERS CLOCK BECAUSE HER FIRST TEACHER HAD TOLD HER THAT A STRUCTURE IS ONLY FINISHED WHEN SOMETHING IRREPLACEABLE HAS CROSSED IT SAFELY AND IN FORTY YEARS SHE NEVER LOST THE CLOCK OR A BRIDGE AND WHEN STUDENTS ASKED HER FOR THE SECRET OF HER CONFIDENCE SHE SAID IT WAS SIMPLY THAT SHE HAD NEVER DESIGNED ANYTHING SHE WOULD NOT PERSONALLY CARRY HER OWN HISTORY ACROSS AND THE CLOCK NOW SITS IN THE ENGINEERING LIBRARY STILL KEEPING TIME",
 "vigenere_nb": "THE VILLAGE BAKER PROVED EVERY LOAF WITH A PIECE OF DOUGH OLDER THAN HIS GRANDFATHER AND CLAIMED THE STARTER REMEMBERED EVERY HARVEST SINCE THE MILL WAS BUILT AND EVERY NEW BAKER IN THE VALLEY BEGGED A SPOONFUL OF IT TO BEGIN A LINE OF THEIR OWN",
 "keyed_columnar": "THE TELEGRAPH OPERATOR LEARNED TO RECOGNIZE EVERY SENDER ON THE LINE BY THE SMALL HESITATIONS BETWEEN LETTERS AND SHE COULD TELL WHO WAS TIRED WHO WAS LYING AND WHO WAS IN LOVE LONG BEFORE THE MESSAGES THEMSELVES ADMITTED ANYTHING OF THE KIND",
 "homophonic_nb": "THE TRAIN DISPATCHER KEPT AN OLD BRASS WHISTLE ON HIS DESK THAT HAD BELONGED TO THE FIRST MAN WHO EVER HELD THE JOB AND HE BLEW IT EXACTLY ONCE EACH YEAR ON THE ANNIVERSARY OF THE NIGHT THE VALLEY FLOODED AND EVERY TRAIN ON THE DIVISION STOPPED FOR ONE FULL MINUTE WHEREVER IT HAPPENED TO BE AND NEW CREWS ALWAYS ASKED WHY AND WERE ALWAYS TOLD THE SAME QUIET STORY ABOUT THE NIGHT THE SCHEDULE MATTERED LESS THAN THE PEOPLE ON IT",
 "bifid_probe": "THE LOCKSMITH COULD OPEN ANYTHING IN THE CITY BUT KEPT ONE RUSTED PADLOCK ON HIS OWN WORKBENCH THAT HE HAD NEVER PICKED BECAUSE HIS TEACHER HAD CLOSED IT ON THE DAY SHE RETIRED",
 "homo_transp_open": "THE DEEP SEA DIVER MEASURED HER CAREER NOT IN YEARS BUT IN MINUTES OF SILENCE BELOW THE REACH OF ANY RADIO AND SHE SPENT THEM ALL LISTENING TO THE HULL SOUNDS OF SHIPS LONG SUNK",
 "padding_composite": "THE STONEMASON SIGNED NOTHING BUT LEFT ONE DELIBERATELY IMPERFECT CHISEL STROKE ON EVERY ARCH HE FINISHED AND A CENTURY LATER THE RESTORERS USED THOSE SMALL FLAWS TO MAP HIS WHOLE WORKING LIFE ACROSS THE CITY BRIDGES CHURCHES AND ONE FORGOTTEN FOUNTAIN AND EVERY SPRING SINCE THE DISCOVERY THE CITY LAYS ONE WHITE FLOWER AT THE FOUNTAIN THAT NOBODY HAD REMEMBERED UNTIL HIS MARKS WERE READ",
 "short_quag": "THE ASTRONOMER NAMED NO COMET AFTER HERSELF AND REGRETTED ONLY THE ONE SHE NEVER REPORTED",
 "misleading_label": "THE BOOKBINDER SEWED A SINGLE RED THREAD INTO THE SPINE OF EVERY BOOK SHE REPAIRED SO THAT A CENTURY OF LIBRARIANS COULD TRACE HER QUIET WORK",
 "railfence_composite": "THE CANAL KEEPER RAISED THE WATER EVERY SPRING BY EXACTLY THE DEPTH OF HIS FATHERS OLD MEASURING STICK AND NOBODY ELSE EVER KNEW THE STICK HAD BEEN BROKEN AND MENDED ONE INCH SHORT A GENERATION AGO SO THE WHOLE VALLEY RAN GENTLY SHALLOW FOR FIFTY YEARS",
 "latin_sub": "PHARMACOPOLA RADICEM AMARAM IN VINO COQUIT ET SUCUM PER LINTEUM COLAT DEINDE AEGRO BIS IN DIE POTUM DAT ET FEBRIS INTRA SEPTEM DIES CEDIT SI AEGER QUIETEM SERVAT ET CIBUM LEVEM SUMIT",
 "german_homophonic": "DER JUNGE GESELLE WANDERTE VON STADT ZU STADT UND SUCHTE EINEN MEISTER DER IHM DIE ALTE KUNST LEHREN WOLLTE UND NACH SIEBEN JAHREN FAND ER EINE KLEINE WERKSTATT IN DER EIN ALTER MANN OHNE WORTE ARBEITETE UND ER BLIEB UND LERNTE DURCH ZUSEHEN MEHR ALS ALLE BUECHER IHM JE GEGEBEN HATTEN",
 "french_sub": "LE VIEUX PECHEUR CONNAISSAIT CHAQUE PIERRE DU PORT ET CHAQUE HUMEUR DE LA MER ET QUAND LES JEUNES PARTAIENT MALGRE LE VENT IL ALLUMAIT UNE LAMPE DANS SA FENETRE ET PERSONNE NE SAVAIT COMBIEN DE BATEAUX CETTE PETITE LUMIERE AVAIT RAMENES",
}

cases = {}


def add(cid, family, tier, expect, cipher, plaintext, extra=None):
    cases[cid] = {"family": family, "tier": tier, "expectation": expect,
                  "ciphertext": cipher, "plaintext_letters": plaintext.replace(" ", ""),
                  "plaintext_spaced": plaintext, **(extra or {})}


def _reroll_composite(pl, width, max_tries=8, min_acc=0.99):
    """Roll substitution+columnar keys until the C.1 peel demonstrably solves."""
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText
    from automated import runner as ar

    last = None
    for attempt in range(1, max_tries + 1):
        m = rand_perm(); kw = rand_kw(width)
        cipher = columnar_encrypt("".join(m[c] for c in pl), kw)
        ct = CipherText(raw=cipher, alphabet=Alphabet.standard_english(), source="gen", separator=None)
        _solver, _key, decryption, _step = ar._run_composite_substitution_transposition(
            ct, language="en", cipher_id="gen_v2")
        acc = sum(a == b for a, b in zip(decryption or "", pl)) / len(pl)
        last = (cipher, kw)
        if acc >= min_acc:
            return cipher, kw, attempt
    raise SystemExit(f"reroll_composite: no key solved width {width} in {max_tries} tries "
                     f"(last acc {acc:.3f}) — lengthen the prose or investigate task #9")


def _reroll_keyed_columnar(pl, width, max_tries=8, min_acc=0.99):
    """Roll plain-columnar keys until solve_transposition demonstrably recovers."""
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText
    from analysis.transposition_solver import solve_transposition

    for attempt in range(1, max_tries + 1):
        kw = rand_kw(width)
        cipher = columnar_encrypt(pl, kw)
        ct = CipherText(raw=cipher, alphabet=Alphabet.standard_english(), source="gen", separator=None)
        # Default-budget bar (60s): the agent's automated_solver experiment runs the
        # solver at its default wall-clock, so a case that needs a bigger budget is
        # not a fair standing-capability case.
        res = solve_transposition(ct, language="en", family_hint="columnar_transposition",
                                  budget_seconds=60.0)
        got = str(res.get("plaintext") or "")
        acc = sum(a == b for a, b in zip(got, pl)) / len(pl)
        if acc >= min_acc:
            return cipher, kw, attempt
    raise SystemExit(f"reroll_keyed_columnar: no key solved width {width} in {max_tries} tries "
                     f"(last acc {acc:.3f}) — see task #8")


def build_cases():
    # gs0 control: mono WITH boundaries.
    p = PROSE["warmup_mono"]; m = rand_perm()
    add("gs0_warmup_mono", "monoalphabetic_substitution", "control", "solve",
        " ".join("".join(m[c] for c in w) for w in p.split()), p, {"word_boundaries": True})

    # gs1/gs2 composite substitution+keyed-columnar, TWO SEEDS (replication).
    # The peel is basin-fragile at some (length, key) combinations (task #9), so
    # standing-capability cases REROLL keys until the peel demonstrably solves —
    # the suite tests the capability where it exists; the fragility has its own task.
    for cid, key_prose, width in (("gs1_composite_subtransp_a", "composite_a", 7),
                                  ("gs2_composite_subtransp_b", "composite_b", 6)):
        p = PROSE[key_prose]; pl = p.replace(" ", "")
        cipher, kw, tries = _reroll_composite(pl, width)
        add(cid, "substitution_transposition", "standing_capability", "solve_in_surface",
            cipher, p,
            {"word_boundaries": False, "columnar_keyword": kw, "reroll_tries": tries,
             "note": f"substitution THEN keyed columnar width {width}; composite experiment; "
                     f"peel-validated at generation ({tries} key trial(s))"})

    # gs3/gs4 blind Quagmire III, TWO SEEDS (replication). kw7/cw8 = default sweep.
    for cid, key_prose in (("gs3_quagmire3_nb_a", "quagmire_a"), ("gs4_quagmire3_nb_b", "quagmire_b")):
        p = PROSE[key_prose]; pl = p.replace(" ", ""); tab = rand_kw(7); cyc = rand_kw(8)
        add(cid, "quagmire3", "standing_capability", "solve_in_surface",
            encode_quagmire_plaintext(pl, cycleword=cyc, quagmire_type="quag3", alphabet_keyword=tab), p,
            {"word_boundaries": False, "alphabet_keyword": tab, "cycleword": cyc,
             "note": "blind Quagmire III; quagmire3_shotgun experiment"})

    # gs5 Vigenère nb — v1 fs5 was blocked by the installer bug (fixed 39b889a): expectation flips to solve.
    p = PROSE["vigenere_nb"]; pl = p.replace(" ", ""); vk = rand_kw(6)
    add("gs5_vigenere_nb", "vigenere", "standing_capability", "solve",
        encode_plaintext(pl, parse_periodic_key(key=vk, variant="vigenere"), variant="vigenere"), p,
        {"word_boundaries": False, "periodic_key": vk,
         "note": "v1 recorded the installer failure (fs5); fixed in 39b889a — a full solve should now "
                 "declare. HINT-DEPENDENT: the blind auto-route hijacks nb periodic to homophonic "
                 "(known F1 gap); the agent must diagnose the period and hint the experiment. The "
                 "hinted periodic engine was verified char 1.000 at generation."})

    # gs6 keyed columnar WIDTH 11 nb — v1 fs3 was an exposure gap (closed 7f3f09f/8e40744): flips to solve_in_surface.
    # Rerolled until solve_transposition demonstrably recovers it (the F2 trigger can
    # false-positive on SA pseudo-English for some keys — task #8).
    p = PROSE["keyed_columnar"]; pl = p.replace(" ", "")
    cipher, kw, tries = _reroll_keyed_columnar(pl, 11)
    add("gs6_keyed_columnar_w11_nb", "keyed_columnar_transposition", "standing_capability", "solve_in_surface",
        cipher, p,
        {"word_boundaries": False, "columnar_keyword": kw, "reroll_tries": tries,
         "note": "width 11 = the F2-escalation regime. HINT-DEPENDENT: the hinted engine "
                 f"(family_hint=columnar) solves at default 60s budget (verified, {tries} key "
                 "trial(s)); the BLIND route fails on this case (task #8) — diagnosing "
                 "transposition and hinting the automated_solver experiment is the agentic "
                 "requirement."})

    # gs7 homophonic nb (numeric homophones), standing frontier.
    p = PROSE["homophonic_nb"]; pl = p.replace(" ", "")
    homo = {}; sym = 1
    for L in string.ascii_uppercase:
        k = rng.choice([1, 1, 2, 2, 3]); homo[L] = [f"{sym+i:02d}" for i in range(k)]; sym += k
    add("gs7_homophonic_nb", "homophonic_substitution", "standing_frontier", "solve_or_strong_partial",
        " ".join(rng.choice(homo[c]) for c in pl), p,
        {"word_boundaries": False, "note": f"{sym-1} numeric homophones; zenith_native"})

    # gs8 Latin mono spaced — v1 measured an honest 0.978 near-miss (shared-symbol repair limit).
    p = PROSE["latin_sub"]; m = rand_perm()
    add("gs8_latin_sub", "simple_substitution", "standing_frontier", "solve_or_strong_partial",
        " ".join("".join(m[c] for c in w) for w in p.split()), p,
        {"language": "la", "word_boundaries": True})

    # gs9 bifid probe (diagnoses_only) — honest-fail discipline; also re-tests the fixed
    # fs7 misroute (a labeled bifid experiment must no longer hijack to the composite peel).
    p = PROSE["bifid_probe"]; pl = p.replace(" ", "").replace("J", "I")
    sq = [c for c in string.ascii_uppercase if c != "J"]; rng.shuffle(sq)
    add("gs9_bifid_probe", "bifid", "honest_fail_probe", "honest_unsolved",
        BifidCipher(period=7).encrypt(pl, "".join(sq)), p.replace("J", "I"),
        {"word_boundaries": False, "note": "fractionation; correct outcome = honest unsolved"})

    # gs10 homophonic + transposition (Z340 class) — open frontier.
    p = PROSE["homo_transp_open"]; pl = p.replace(" ", "")
    hk = {}; s2 = 1
    for L in string.ascii_uppercase:
        k = rng.choice([1, 2]); hk[L] = [f"{s2+i:02d}" for i in range(k)]; s2 += k
    homo_tokens = [rng.choice(hk[c]) for c in pl]
    pipe = TransformPipeline.from_raw({"steps": [{"name": "MatrixRotate", "data": {"width": 13, "direction": "cw"}}]})
    n = (len(homo_tokens) // 13) * 13
    add("gs10_homo_transp_open", "transposition_homophonic", "open_frontier", "honest_unsolved_or_partial",
        " ".join(make_inverse_input_for_pipeline(homo_tokens[:n], pipe)), p,
        {"word_boundaries": False, "note": "homophonic THEN transposition — genuine open frontier"})

    # gs11 PADDING-TAIL composite (declaration probe): plaintext padded to a full
    # grid with trailing M run; a perfect crack carries the tail and must be
    # declared via the rank-2 trimmed candidate (P2, 39b889a).
    p = PROSE["padding_composite"]; pl = p.replace(" ", "")
    pad = (-len(pl)) % 7 or 7
    pad = pad if pad >= 4 else pad + 7
    pl_padded = pl + "M" * pad
    cipher, kw, tries = _reroll_composite(pl_padded, 7)
    add("gs11_padding_composite", "substitution_transposition", "declaration_probe",
        "solve_in_surface_declared_trimmed",
        cipher, p,
        {"word_boundaries": False, "columnar_keyword": kw, "padding": {"char": "M", "length": pad},
         "plaintext_letters_padded": pl_padded, "reroll_tries": tries,
         "note": "grid-padding tail; the FULL decode ends in a meaningless M-run — the "
                 "declarable reading is the rank-2 padding-trimmed candidate; peel-validated "
                 f"at generation ({tries} key trial(s))"})

    # gs12 SHORT quagmire3 (~90 letters) — evidence-restraint regime.
    p = PROSE["short_quag"]; pl = p.replace(" ", ""); tab = rand_kw(7); cyc = rand_kw(8)
    add("gs12_short_quag3_nb", "quagmire3", "short_text_probe", "solve_in_surface_or_honest",
        encode_quagmire_plaintext(pl, cycleword=cyc, quagmire_type="quag3", alphabet_keyword=tab), p,
        {"word_boundaries": False, "alphabet_keyword": tab, "cycleword": cyc,
         "note": "short-text regime; a confident wrong declaration is the failure mode being probed"})

    # gs13 MISLEADING-LABEL probe: plain mono nb, but the permitted context asserts
    # a Vigenère with a six-letter key. Evidence must beat the anchor.
    p = PROSE["misleading_label"]; pl = p.replace(" ", ""); m = rand_perm()
    add("gs13_misleading_label_nb", "monoalphabetic_substitution", "anchor_probe", "solve_overriding_label",
        "".join(m[c] for c in pl), p,
        {"word_boundaries": False, "anchor_trap": True,
         "supplied_context": "An archivist's accession note claims this is a Vigenere cipher with a six-letter key.",
         "note": "the supplied context is WRONG by design; IC/periodic evidence says mono"})

    # gs14 German homophonic nb — language-capability probe (non-en homophonic is a tracked gap).
    p = PROSE["german_homophonic"]; pl = p.replace(" ", "")
    gh = {}; s3 = 1
    for L in string.ascii_uppercase:
        k = rng.choice([1, 1, 2]); gh[L] = [f"{s3+i:02d}" for i in range(k)]; s3 += k
    add("gs14_german_homophonic_nb", "homophonic_substitution", "language_probe", "partial_or_honest",
        " ".join(rng.choice(gh[c]) for c in pl), p,
        {"language": "de", "word_boundaries": False,
         "note": "German homophonic; the zenith path is en-tuned — measures the de gap honestly"})

    # gs15 French mono spaced — language-capability probe. MEASURED at generation:
    # the blind automated route MISROUTES fr mono into the composite peel (task #9),
    # so the automated floor is ~0.08; the agent surface (mono search + fr dicts +
    # repair) is what is being probed. Expectation set accordingly.
    p = PROSE["french_sub"]; m = rand_perm()
    add("gs15_french_sub", "simple_substitution", "language_probe", "partial_or_solve",
        " ".join("".join(m[c] for c in w) for w in p.split()), p,
        {"language": "fr", "word_boundaries": True,
         "note": "fr models/dicts exist; the blind content route misroutes to the composite "
                 "peel (measured 0.08) — the agent must drive mono search directly"})

    # gs16 railfence composite — MEASURED at generation: the composite peel enters and
    # reports peeled_and_solved but lands wrong-basin garbage (0.09) — the geometric
    # arm does not cover rail-3 (task #9). Recorded honestly as an open-frontier probe.
    p = PROSE["railfence_composite"]; pl = p.replace(" ", ""); m = rand_perm()
    add("gs16_railfence_composite", "substitution_transposition", "open_frontier", "partial_or_honest",
        RailfenceCipher().encrypt("".join(m[c] for c in pl), 3), p,
        {"word_boundaries": False, "transposition": "railfence3",
         "note": "substitution THEN 3-rail fence; measured: the peel false-positives into "
                 "garbage here — correct agent outcome is honest unsolved (or a transform-"
                 "screen rescue); a false declaration is the failure being probed"})

    return cases


def add_borg_case():
    """gs17: one historical Borg Track-B page (canonical S-token transcription)."""
    try:
        from benchmark.loader import BenchmarkLoader
        root = Path.home() / "Dropbox/src2/cipher_benchmark/benchmark"
        loader = BenchmarkLoader(str(root))
        tests = loader.load_tests("borg_tests.jsonl", track="transcription2plaintext")
        target = next(t for t in tests if t.test_id == "borg_single_B_borg_0077v")
        data = loader.load_test_data(target)
        cases["gs17_borg_historical"] = {
            "family": "monoalphabetic_substitution_historical", "tier": "historical",
            "expectation": "solve_or_strong_partial",
            "ciphertext": data.canonical_transcription,
            "plaintext_letters": (data.plaintext or "").replace(" ", ""),
            "plaintext_spaced": data.plaintext or "",
            "language": "la", "word_boundaries": True, "format": "canonical",
            "benchmark_test_id": "borg_single_B_borg_0077v",
            "note": "historical Borg page, canonical S-tokens; GT lives in the benchmark repo",
        }
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: borg case skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
        return False


DOC_HEADER = """# Decipher — Agentic Frontier Suite v2 (2026-07-20)

18 fresh ciphers for a single Codex/Sol session to crack via the Decipher MCP
server. Supersedes the 2026-07-19 v1 suite (all v1 plaintexts are burned —
full decodes appear in the graded results doc). Contamination-free original
prose; families deliberately NOT stated (diagnosis is part of each crack).
New in v2: replication seeds for the flagship families, post-fix expectation
flips, and probe classes for declaration (padding tails), short-text
restraint, misleading context, non-English capability, and one historical
page.

## Setup (once, before starting)
1. In this repo: `git pull --ff-only && sh scripts/bootstrap.sh`, then restart the app.
2. Confirm current code: `investigation_list` `server_code.git_head` should match
   `git rev-parse --short HEAD`, and `experiment_submit` should advertise both
   `quagmire3_shotgun` and `composite_substitution_transposition`. If not, the
   server is stale — restart and re-check.

## Orchestration — ONE SUBAGENT PER CIPHER (18 total, run them in parallel)
For EACH cipher below, spin up a dedicated sub-agent whose sole task is to crack
that one cipher through the Decipher MCP tools. Each sub-agent:
- calls `investigation_start` with the ciphertext inline + the stated language
  (use format `canonical` where a case says so);
- drives the investigation per `docs/mcp_onboarding.md` §Investigation methodology,
  from `investigation_status` (diagnose family -> hypothesis branches ->
  experiments -> read -> verify);
- uses ONLY Decipher MCP tools — no repo scripts/solvers/Rust directly (this
  measures the tool surface; record honestly if the surface cannot do something);
- treats any supplied accession/context note as a CLAIM, not ground truth;
- honors WF-7: before stopping, `request_independent_verification` on the leading
  branch, then close with `meta_declare_solution` or `meta_declare_unsolved`.

## Result format — each sub-agent returns EXACTLY this block
```
### <cipher_id>
- investigation_id: <id>
- verdict: solved | unsolved
- family (your diagnosis): <e.g. monoalphabetic / homophonic / periodic-poly / ...>
- leading branch: <name>
- decode (decode_show, FULL — even if partial/damaged):
<the decoded text>
- signals: dict_rate=<>, quad=<>
- attestation: accepts=<bool>, coherence=<0-10>, language_conf=<>, recoverability=<>, damage=<>
- route: <one paragraph: how you diagnosed and solved/failed; any tool the surface lacked>
```
Do not read files under `docs/evidence/`, `artifacts/`, or anywhere in
`~/Dropbox/src2/cipher_benchmark/` during the run.

---

## Ciphers
"""


def write_doc(ordered_ids):
    lines = [DOC_HEADER]
    for i, cid in enumerate(ordered_ids, 1):
        c = cases[cid]
        lang = c.get("language", "en")
        wb = "word boundaries present (spaced)" if c.get("word_boundaries") else "NO word boundaries (single continuous stream)"
        ctx = f"language `{lang}`; {wb}"
        if c.get("format") == "canonical":
            ctx += "; canonical S-token transcription (` | ` separates words) — use format `canonical`"
        extra = ""
        if c.get("supplied_context"):
            extra = f"\nSupplied context (treat as a CLAIM, not ground truth): {c['supplied_context']}"
        lines.append(f"### Cipher {i} — id `{cid}`\nPermitted context: {ctx}.{extra}\n\n```\n{c['ciphertext']}\n```\n")
    out = REPO / "docs/evidence/agentic_frontier_suite_v2.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def seal_answers():
    out = Path.home() / ".config/decipher/dogfood_answers/agentic_frontier_answers_v2.json"
    out.write_text(json.dumps(
        {"generator": "gen_agentic_frontier_suite_v2.py seed=0xBEEF", "cases": cases},
        indent=2, default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="run the $0 local behavior checks")
    args = ap.parse_args()

    build_cases()
    add_borg_case()
    ordered = list(cases)
    doc = write_doc(ordered)
    sealed = seal_answers()
    for cid, c in cases.items():
        print(f"{cid:30s} {c['tier']:22s} {c['expectation']:34s} len={len(c['ciphertext'])}")
    print(f"\ndoc    -> {doc}\nsealed -> {sealed}")

    if args.validate:
        run_validation()


def run_validation():
    """$0 checks: measure each synthetic case's behavior on the capable engine."""
    import time
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText
    from automated import runner as ar
    from analysis.polyalphabetic_fast import search_quagmire3_shotgun_fast

    def char_acc(got, want):
        got = (got or "").replace(" ", ""); want = want.replace(" ", "")
        if not want:
            return 0.0
        return sum(a == b for a, b in zip(got, want)) / len(want)

    def letters_ct(raw, lang="en"):
        return CipherText(raw=raw, alphabet=Alphabet.standard_english(), source="v2", separator=None)

    def numeric_ct(raw):
        toks = raw.split()
        alpha = Alphabet(sorted(set(toks)))
        return CipherText(raw=raw, alphabet=alpha, source="v2", separator=" ")

    print("\n--- validation ($0 local) ---")
    for cid, c in cases.items():
        if c["tier"] == "historical":
            print(f"{cid:30s} SKIP (benchmark GT known)")
            continue
        want = c.get("plaintext_letters_padded") or c["plaintext_letters"]
        lang = c.get("language", "en")
        t0 = time.time()
        try:
            fam = c["family"]
            if fam == "quagmire3":
                res = search_quagmire3_shotgun_fast(
                    letters_ct(c["ciphertext"]), language="en",
                    keyword_lengths=[7], cycleword_lengths=[8],
                    hillclimbs=5000, restarts=250, seed=1, top_n=3, threads=0)
                got = (res.get("top_candidates") or [{}])[0].get("plaintext", "")
            elif fam in ("homophonic_substitution", "transposition_homophonic"):
                r = ar.run_automated(numeric_ct(c["ciphertext"]), lang)
                got = r.final_decryption or ""
            else:
                raw = c["ciphertext"].replace(" ", "") if not c.get("word_boundaries") else c["ciphertext"]
                ct = CipherText(raw=raw, alphabet=Alphabet.standard_english(), source="v2",
                                separator=None if not c.get("word_boundaries") else " ")
                r = ar.run_automated(ct, lang)
                got = r.final_decryption or ""
            acc = char_acc(got, want)
            print(f"{cid:30s} char={acc:0.3f}  [{time.time()-t0:5.1f}s]  expect={c['expectation']}")
        except Exception as exc:  # noqa: BLE001
            print(f"{cid:30s} ERROR {type(exc).__name__}: {str(exc)[:90]}")


if __name__ == "__main__":
    main()
