"""Generate the pasteable agentic frontier suite: fresh contamination-free
ciphers spanning the (recently extended) frontier, with sealed answers and a
$0 local verification of each case's expected behavior.

Ciphertexts + the orchestration brief -> docs/evidence/agentic_frontier_suite.md
Sealed answers -> ~/.config/decipher/dogfood_answers/agentic_frontier_answers.json
"""
import json, random, string, sys, time
from pathlib import Path

REPO = Path.home() / "Dropbox/src2/decipher"
sys.path.insert(0, str(REPO / "src"))

from analysis.polyalphabetic import encode_plaintext, encode_quagmire_plaintext, parse_periodic_key
from ciphers.transposition import columnar_encrypt
from ciphers.fractionation import BifidCipher
from analysis.transformers import TransformPipeline, make_inverse_input_for_pipeline

rng = random.Random(0xF00D)

def rand_perm():
    p = list(string.ascii_uppercase); rng.shuffle(p); return dict(zip(string.ascii_uppercase, p))
def rand_kw(n):
    letters = list(string.ascii_uppercase); rng.shuffle(letters); return "".join(letters[:n])

# ---- fresh original prose (contamination-free), one per case ----
PROSE = {
 "warmup_mono": "THE OLD CLOCKMAKER LEFT EACH FINISHED PIECE RUNNING ONE MINUTE FAST SO THE TOWN WOULD NEVER BE LATE FOR ANYTHING THAT TRULY MATTERED",
 "composite_subtransp": "THE CARTOGRAPHER DREW COASTLINES SHE HAD NEVER SEEN FROM THE STORIES OF SAILORS AND WHEN ONE MAP PROVED EXACTLY RIGHT SHE BURNED IT SO NO NAVY WOULD EVER FIND THE ISLAND SHE HAD INVENTED BY ACCIDENT AND THEN MADE REAL AND FOR THE REST OF HER LIFE SHE WONDERED WHETHER THE OTHER PLACES ON HER FALSE MAPS WERE ALSO WAITING QUIETLY TO BE DISCOVERED BY THE FIRST SHIP THAT BELIEVED IN THEM ENOUGH TO SET A COURSE",
 "quagmire3": "THE BEEKEEPER TAUGHT HER APPRENTICE THAT A CALM HAND MATTERS MORE THAN A CLEVER ONE AND THAT THE HIVE FORGIVES A SLOW MISTAKE FAR SOONER THAN A QUICK CORRECTION AND SO THE BOY LEARNED PATIENCE LONG BEFORE HE EVER LEARNED THE CRAFT AND WHEN HE FINALLY KEPT HIVES OF HIS OWN HE FOUND THAT EVERY LESSON SHE HAD GIVEN HIM WAS REALLY ABOUT LISTENING TO SMALL SIGNALS AND TRUSTING THAT THE COLONY KNEW THINGS THE KEEPER COULD ONLY GUESS AT AND HE PASSED THE SAME QUIET WISDOM TO HIS OWN STUDENTS EACH SPRING WHEN THE FIRST WARM DAY WOKE THE SLEEPING FRAMES",
 "keyed_columnar": "THE NIGHT LIBRARIAN SHELVED RETURNED BOOKS BY THE WARMTH STILL LEFT IN THEIR COVERS AND CLAIMED SHE COULD TELL WHICH ONES HAD BEEN READ BESIDE A FIRE AND WHICH HAD ONLY BEEN CARRIED HOME AND LEFT UNOPENED ON A SHELF UNTIL THE BORROWER FELT GUILTY ENOUGH TO RETURN THEM UNTOUCHED AND SLIGHTLY ASHAMED",
 "homophonic_nb": "THE TIDE CLERK MEASURED THE HARBOR TWICE EACH DAY AND KEPT A PRIVATE COLUMN FOR THE MORNINGS WHEN THE WATER ROSE HIGHER THAN THE TABLES SAID IT SHOULD AND AFTER THIRTY YEARS THAT COLUMN PREDICTED STORMS BETTER THAN THE BAROMETER AND WHEN HE RETIRED HE LEFT THE NOTEBOOK TO A YOUNG CLERK WHO DID NOT BELIEVE IN IT UNTIL THE WINTER THE WHOLE COAST FLOODED AND ONLY THAT ONE STUBBORN COLUMN HAD SEEN IT COMING WEEKS AHEAD OF EVERY INSTRUMENT",
 "vigenere_nb": "THE RETIRED SURVEYOR WALKED THE SAME RIDGE EVERY AUTUMN TO CHECK WHETHER THE MOUNTAIN HAD MOVED AND EACH YEAR HE WROTE UNCHANGED IN HIS FIELD BOOK WITH A SATISFACTION THAT PUZZLED EVERYONE WHO WAS NOT ALSO A SURVEYOR",
 "latin_sub": "MEDICUS HERBAM SICCAM CUM OLEO MISCET ET UNGUENTUM PARAT DEINDE VULNUS LENITER TEGIT ET AEGRUM QUIETEM SERVARE IUBET POST TRES DIES TUMOR CEDIT ET CARO NOVA CRESCIT",
 "bifid_probe": "THE MUSEUM GUARD MEMORIZED EVERY CREAK OF THE OLD PARQUET FLOOR AND COULD NAME ANY INTRUDER BY THEIR WEIGHT ALONE LONG BEFORE THE CAMERAS EVER FOUND THEM MOVING THROUGH THE DARKENED GALLERIES",
 "homo_transp_open": "THE RIVER PILOT REFUSED TO TRUST ANY CHART DRAWN BY SOMEONE WHO HAD NOT DROWNED AT LEAST ONCE AND LIVED TO REDRAW IT AND SO HIS OWN MAPS CARRIED A SMALL CROSS AT EVERY PLACE THE RIVER HAD TRIED TO KILL HIM",
}

cases = {}
def add(cid, family, tier, expect, cipher, plaintext, extra=None):
    cases[cid] = {"family": family, "tier": tier, "expectation": expect,
                  "ciphertext": cipher, "plaintext_letters": plaintext.replace(" ",""),
                  "plaintext_spaced": plaintext, **(extra or {})}

# 0. warmup: mono substitution WITH boundaries (should trivially solve)
p = PROSE["warmup_mono"]; m = rand_perm()
add("fs0_warmup_mono","monoalphabetic_substitution","control","solve",
    " ".join("".join(m[c] for c in w) for w in p.split()), p, {"word_boundaries":True})

# 1. composite substitution + columnar (round-4 class, no boundaries) — EXTENDED FRONTIER
p = PROSE["composite_subtransp"]; pl = p.replace(" ",""); m = rand_perm(); kw = rand_kw(7)
add("fs1_composite_subtransp","substitution_transposition","extended_frontier","solve_in_surface",
    columnar_encrypt("".join(m[c] for c in pl), kw), p, {"word_boundaries":False,"note":"substitution THEN columnar; peel-and-solve + composite experiment"})

# 2. blind Quagmire III no-boundary (round-6 class) — EXTENDED FRONTIER
p = PROSE["quagmire3"]; pl = p.replace(" ",""); tab = rand_kw(7); cyc = rand_kw(8)
add("fs2_quagmire3_nb","quagmire3","extended_frontier","solve_in_surface",
    encode_quagmire_plaintext(pl, cycleword=cyc, quagmire_type="quag3", alphabet_keyword=tab), p,
    {"word_boundaries":False,"note":"blind Quagmire III; quagmire3_shotgun experiment"})

# 3. keyed columnar transposition no-boundary — OPEN FRONTIER (gap: exposure)
# The F2 columnar_search module solves this 100%, but it lives ONLY inside the
# composite peel (_peel_order_layer). A PLAIN keyed columnar auto-routes to the
# geometric transposition screen, which F2 proved fails on keyed columnar. So
# this is a real gap: capable module, not exposed to the agent standalone.
p = PROSE["keyed_columnar"]; pl = p.replace(" ",""); kw = rand_kw(6)
add("fs3_keyed_columnar_nb","keyed_columnar_transposition","open_frontier","honest_unsolved_or_escape",
    columnar_encrypt(pl, kw), p, {"word_boundaries":False,"note":"keyed columnar. F2 module solves it (100%) but only inside the composite peel; the standalone transposition route uses the geometric screen (fails). GAP = expose columnar_search as a standalone route."})

# 4. no-boundary homophonic (standing frontier)
p = PROSE["homophonic_nb"]; pl = p.replace(" ","")
homo_perm = {}; sym = 1
for L in string.ascii_uppercase:  # 1-3 homophones per letter by frequency-ish
    k = rng.choice([1,1,2,2,3]); homo_perm[L] = [f"{sym+i:02d}" for i in range(k)]; sym += k
homo_cipher = " ".join(rng.choice(homo_perm[c]) for c in pl)
add("fs4_homophonic_nb","homophonic_substitution","standing_frontier","solve_or_strong_partial",
    homo_cipher, p, {"word_boundaries":False,"note":f"{sym-1} symbols, numeric homophones; zenith_native"})

# 5. Vigenere no-boundary (standing; agent must diagnose periodic)
p = PROSE["vigenere_nb"]; pl = p.replace(" ",""); vk = rand_kw(6)
add("fs5_vigenere_nb","vigenere","standing_frontier","solve",
    encode_plaintext(pl, parse_periodic_key(key=vk, variant="vigenere"), variant="vigenere"), p,
    {"word_boundaries":False})

# 6. Latin simple substitution WITH boundaries (non-English routing)
p = PROSE["latin_sub"]; m = rand_perm()
add("fs6_latin_sub","simple_substitution","standing_frontier","solve",
    " ".join("".join(m[c] for c in w) for w in p.split()), p, {"language":"la","word_boundaries":True})

# 7. Bifid period 7 (diagnoses_only) — HONEST-FAIL / anchoring-trap probe
p = PROSE["bifid_probe"]; pl = p.replace(" ",""); sq = [c for c in string.ascii_uppercase if c!="J"]; rng.shuffle(sq)
add("fs7_bifid_probe","bifid","honest_fail_probe","honest_unsolved",
    BifidCipher(period=7).encrypt(pl, "".join(sq)), p,
    {"word_boundaries":False,"note":"fractionation; diagnosis mis-calls mono (F3). Correct outcome = honest unsolved."})

# 8. homophonic + transposition no-boundary (Z340-class) — OPEN FRONTIER
p = PROSE["homo_transp_open"]; pl = p.replace(" ","")
hk = {}; s2 = 1
for L in string.ascii_uppercase:
    k = rng.choice([1,2]); hk[L] = [f"{s2+i:02d}" for i in range(k)]; s2 += k
homo_tokens = [rng.choice(hk[c]) for c in pl]
pipe = TransformPipeline.from_raw({"steps":[{"name":"MatrixRotate","data":{"width":13,"direction":"cw"}}]})
n = (len(homo_tokens)//13)*13
scrambled = make_inverse_input_for_pipeline(homo_tokens[:n], pipe)
add("fs8_homo_transp_open","transposition_homophonic","open_frontier","honest_unsolved_or_partial",
    " ".join(scrambled), p, {"word_boundaries":False,"note":"homophonic THEN transposition (Z340-class) — genuine open frontier."})

# ---- seal answers, write ciphertext count ----
out = Path.home()/".config/decipher/dogfood_answers/agentic_frontier_answers.json"
out.write_text(json.dumps({"generator":"gen_agentic_frontier.py seed=0xF00D","cases":cases}, indent=2, default=str))
for cid,c in cases.items():
    print(f"{cid:28s} {c['tier']:20s} {c['expectation']:26s} len={len(c['ciphertext'])}")
print(f"\nsealed -> {out}")
