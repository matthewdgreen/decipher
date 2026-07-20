# Agentic Frontier Suite — Codex Results + Grading (2026-07-19)

Codex ran `docs/evidence/agentic_frontier_suite.md` (9 cases, one sub-agent
each) at clean commit `af73cd1`. Graded vs the sealed answers
(`~/.config/decipher/dogfood_answers/agentic_frontier_answers.json`). Raw
per-case report is appended below (verbatim from the run).

## Graded summary

| case | verdict | char | expectation | read |
|---|---|---|---|---|
| fs0_warmup_mono | solved | 1.000 | solve | ✓ clean solve, declared |
| fs1_composite_subtransp | **solved** | **1.000** | solve_in_surface | **✓ EXTENDED-FRONTIER WIN** — composite peel + experiment, declared coh 10 |
| fs2_quagmire3_nb | **solved** | **1.000** | solve_in_surface | **✓ EXTENDED-FRONTIER WIN** — quagmire3_shotgun (64×50k), declared coh 9 |
| fs5_vigenere_nb | unsolved | n/a | solve | **BUG** — experiment recovered the FULL plaintext (key IHBMEP) but the periodic-experiment INSTALLER didn't materialize the key state, so `decode_show` showed `????` and the gate rejected. Solve succeeded; declaration blocked by an installer defect. |
| fs6_latin_sub | unsolved | 0.978 | solve | near-miss; batch repair fixed CUIETEM→QUIETEM but IUBET couldn't be fixed (shares a cipher symbol with TERRAM); honest unsolved. Repair-primitive shared-symbol limitation. |
| fs4_homophonic_nb | unsolved | 0.994 | solve/partial | near-miss; verifier-arbitrated repair fixed CLERK/KEPT/YOUNG; last residual THETINTER→THEWINTER is globally COUPLED (also flips ANDTHENHE→ANDWHENHE), evidence gate rejected the coupled edit; honest unsolved. |
| fs3_keyed_columnar_nb | unsolved | 0.069 | honest/escape | **✓ VALIDATES THE F2 EXPOSURE GAP** — agent ran the geometric transform screen (fails on keyed columnar); the F2 `search_keyed_columnar` module is walled inside the composite peel, unreachable standalone. Honest unsolved. **CORRECTION (2026-07-19):** the standalone route has covered keyed columnar since `8e40744` (via `solve_transposition`'s SA column-order search; `run_automated` solves this fs3 ciphertext directly). This row's failure was agent-surface tool-selection (geometric transform screen instead of an `automated_solver` experiment), not a missing capability; the measured residual was width-11 SA misses, now closed by the `keyed_columnar_f2` escalation in `solve_transposition`. |
| fs7_bifid_probe | unsolved | 0.075 | honest | ✓ F3 anchoring trap held. Also: `cipher_system=bifid` was mis-redirected by the misroute guard to composite peeling (minor guard bug). Honest unsolved. |
| fs8_homo_transp_open | unsolved | 0.067 | honest/partial | ✓ open frontier (Z340-class); honest unsolved. |

## Findings
1. **The extended frontier works in-surface.** Composite sub+transposition and
   blind Quagmire III both SOLVED and DECLARED on a fresh Codex clone (1.0, coh
   9–10) via the new experiments. Headline validation of the composite program +
   quagmire3_shotgun.
2. **Discipline held perfectly.** All 6 non-clean cases closed WF-7-honest; zero
   over-declaration. The 3 open-frontier/probe cases (fs3/fs7/fs8) failed exactly
   as predicted.
3. **A "solve succeeds, declaration blocked" CLASS across 3 families** — this
   extends workplan step 1 beyond composite padding:
   - fs5 (NEW BUG): the periodic-polyalphabetic experiment installer does not
     materialize the key onto the branch, so a fully-recovered Vigenère renders
     as `????` and can't be verified/declared. Same shape as the composite
     "no base key" gap, for periodic. Actionable installer fix.
   - fs6/fs4: repair primitives can't resolve shared-cipher-symbol (Latin) or
     globally-coupled-homophone (homophonic) edits — a near-perfect solve stalls
     one position short of a clean declaration.
4. **fs3 validates the keyed-columnar exposure gap** (workplan step 3): capable
   module, not reachable standalone.
5. **Minor: the misroute guard redirects `cipher_system=bifid` to the composite
   type** (fs7). Bifid isn't a sub+transposition composite — tighten the guard.

Investigation ids: fs0 f50de91fd7be, fs1 af32ff70de28, fs2 eb4c12809780,
fs5 eab74802ecec, fs6 023ffd1106d5, fs4 9701e3448ea5, fs3 1bce5f091a23,
fs7 43ca1ea1f3c3, fs8 a55d62a33310.

---

# Decipher — Agentic Frontier Suite Results (2026-07-19)

Executed through the Decipher MCP server at clean commit `af73cd1`, using one
dedicated subagent per cipher. Each investigation was independently verified
and explicitly closed under WF-7. Results below retain the suite's required
reporting format.

### fs0_warmup_mono
- investigation_id: f50de91fd7be
- verdict: solved
- family (your diagnosis): monoalphabetic_substitution
- leading branch: mono_solved
- decode (decode_show, FULL — even if partial/damaged):
THE OLD CLOCKMAKER LEFT EACH FINISHED PIECE RUNNING ONE MINUTE FAST SO THE TOWN WOULD NEVER BE LATE FOR ANYTHING THAT TRULY MATTERED
- signals: dict_rate=0.913, quad=-4.9028
- attestation: accepts=true, coherence=10, language_conf=1.0, recoverability=1.0, damage=local
- route: Diagnosis strongly favored monoalphabetic substitution from the near-English IC, peaked frequency shape, and displaced monograms. I created an explicit monoalphabetic hypothesis branch, ran the automated monoalphabetic solver, installed its top candidate as `mono_solved`, inspected the complete decode, obtained positive independent verification, and declared the solution.

### fs5_vigenere_nb
- investigation_id: eab74802ecec
- verdict: unsolved
- family (your diagnosis): periodic Vigenere, fundamental period 6
- leading branch: vigenere_p6_solved
- decode (decode_show, FULL — even if partial/damaged):
???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
- signals: dict_rate=0.0, quad=null
- attestation: accepts=false, coherence=0, language_conf=0.0, recoverability=0.0, damage=basin_wide
- route: Diagnosis strongly identified periodic polyalphabetic encryption from depressed global IC and significant period-6 IC recovery. The automated Vigenere experiment recovered key IHBMEP and the complete plaintext “THERETIREDSURVEYORWALKEDTHESAMERIDGEEVERYAUTUMNTOCHECKWHETHERTHEMOUNTAINHADMOVEDANDEACHYEARHEWROTEUNCHANGEDINHISFIELDBOOKWITHASATISFACTIONTHATPUZZLEDEVERYONEWHOWASNOTALSOASURVEYOR”; however, the experiment installer did not materialize its periodic key state, so decode_show exposed only placeholders and independent verification rejected that rendered branch. WF-7 was closed explicitly as unsolved under the enforced attestation policy.

### fs1_composite_subtransp
- investigation_id: af32ff70de28
- verdict: solved
- family (your diagnosis): substitution_transposition — monoalphabetic substitution followed by 7-column columnar transposition
- leading branch: composite_solved
- decode (decode_show, FULL — even if partial/damaged):
THECARTOGRAPHERDREWCOASTLINESSHEHADNEVERSEENFROMTHESTORIESOFSAILORSANDWHENONEMAPPROVEDEXACTLYRIGHTSHEBURNEDITSONONAVYWOULDEVERFINDTHEISLANDSHEHADINVENTEDBYACCIDENTANDTHENMADEREALANDFORTHERESTOFHERLIFESHEWONDEREDWHETHERTHEOTHERPLACESONHERFALSEMAPSWEREALSOWAITINGQUIETLYTOBEDISCOVEREDBYTHEFIRSTSHIPTHATBELIEVEDINTHEMENOUGHTOSETACOURSE
- signals: dict_rate=0.9398, quad=-5.3823
- attestation: accepts=true, coherence=10, language_conf=1.0, recoverability=1.0, damage=local
- route: Diagnosis found displaced monograms and language-like IC but absent adjacency, ranking substitution_transposition first; I created the `composite_peel` hypothesis and ran the advertised `composite_substitution_transposition` C.1 peel-and-solve experiment, which exhaustively recovered a 7-column columnar layer (order 2,4,0,5,6,1,3; keyword CFAGBDE) and then solved the monoalphabetic substitution. I installed the result as `composite_solved`, read the full decode, compared it against `main`, obtained a fresh independent attestation accepting it at coherence 10/10, declared the solution, and confirmed the investigation explicitly closed with terminal status `solved` at revision 10.

### fs6_latin_sub
- investigation_id: 023ffd1106d5
- verdict: unsolved
- family (your diagnosis): monoalphabetic substitution
- leading branch: latin_repaired
- decode (decode_show, FULL — even if partial/damaged):
MEDICUS TERRAM SICCAM CUM OLEO MISCET ET UNGUENTUM PARAT DEINDE VULNUS LENITER TEGIT ET AEGRUM QUIETEM SERVARE IURET POST TRES DIES TUMOR CEDIT ET CARO NOVA CRESCIT
- signals: dict_rate=0.7037, quad=-4.9101
- attestation: accepts=true, coherence=9, language_conf=0.99, recoverability=0.97, damage=local
- route: Diagnosis strongly favored monoalphabetic substitution; a native substitution experiment recovered a sustained Latin medical passage. Independent verification identified CUIETEM and IURET as local anomalies. Batch repair safely installed CUIETEM→QUIETEM, but testing IURET→IUBET caused collateral TERRAM→TERBAM because both positions share one cipher symbol. Reverification accepted the text as genuine Latin but not as a complete solution, so WF-7 was closed explicitly as unsolved with one irreducible position-specific anomaly.

### fs2_quagmire3_nb
- investigation_id: eb4c12809780
- verdict: solved
- family (your diagnosis): Quagmire III keyed-tableau periodic polyalphabetic, period 8
- leading branch: quagmire3_solved
- decode (decode_show, FULL — even if partial/damaged):
THEBEEKEEPERTAUGHTHERAPPRENTICETHATACALMHANDMATTERSMORETHANACLEVERONEANDTHATTHEHIVEFORGIVESASLOWMISTAKEFARSOONERTHANAQUICKCORRECTIONANDSOTHEBOYLEARNEDPATIENCELONGBEFOREHEEVERLEARNEDTHECRAFTANDWHENHEFINALLYKEPTHIVESOFHISOWNHEFOUNDTHATEVERYLESSONSHEHADGIVENHIMWASREALLYABOUTLISTENINGTOSMALLSIGNALSANDTRUSTINGTHATTHECOLONYKNEWTHINGSTHEKEEPERCOULDONLYGUESSATANDHEPASSEDTHESAMEQUIETWISDOMTOHISOWNSTUDENTSEACHSPRINGWHENTHEFIRSTWARMDAYWOKETHESLEEPINGFRAMES
- signals: dict_rate=0.931, quad=-5.5825
- attestation: accepts=true, coherence=9, language_conf=0.99, recoverability=0.97, damage=local
- route: Diagnosis found near-random global IC 0.0408 but strong period-8 stream recovery to IC 0.0706, supporting periodic polyalphabetic encryption. A recorded keyed-tableau hypothesis was tested with Quagmire III shotgun search at cycleword length 8; shallow and initial deep runs found local basins, while a 64-restart, 50,000-hillclimb search converged four top finalists to the same complete English plaintext. The installed branch was read, compared against main, independently verified as a complete solution, and explicitly closed solved under WF-7.

### fs4_homophonic_nb
- investigation_id: 9701e3448ea5
- verdict: unsolved
- family (your diagnosis): homophonic substitution
- leading branch: semantic_repair
- decode (decode_show, FULL — even if partial/damaged):
THETIDECLERKMEASUREDTHEHARBORTWICEEACHDAYANDKEPTAPRIVATECOLUMNFORTHEMORNINGSWHENTHEWATERROSEHIGHERTHANTHETABLESSAIDITSHOULDANDAFTERTHIRTYYEARSTHATCOLUMNPREDICTEDSTORMSBETTERTHANTHEBAROMETERANDTHENHERETIREDHELEFTTHENOTEBOOKTOAYOUNGCLERKWHODIDNOTBELIEVEINITUNTILTHETINTERTHEWHOLECOASTFLOODEDANDONLYTHATONESTUBBORNCOLUMNHADSEENITCOMINGWEEKSAHEADOFEVERYINSTRUMENT
- signals: dict_rate=0.8864, quad=-5.5385
- attestation: accepts=false, coherence=9, language_conf=0.99, recoverability=0.95, damage=local
- route: The flat 40-symbol, 359-token no-boundary fingerprint supported homophonic substitution, so I ran the full Zenith-native automated route and a two-stage confirmation, then recorded a reading and installed a verifier-arbitrated semantic repair that corrected CLERK, KEPT, and YOUNG. A targeted refinement converged to the same basin; the sole residual `THETINTER` is almost certainly `THEWINTER`, but changing its T-homophone to W also changes `ANDTHENHE` to the plausible `ANDWHENHE`. The repair evidence gate rejected that globally coupled edit, and two independent attestations accepted the text as genuine fluent English but not as a complete solution, so I explicitly closed under WF-7 as unsolved.

### fs3_keyed_columnar_nb
- investigation_id: 1bce5f091a23
- verdict: unsolved
- family (your diagnosis): pure keyed columnar transposition
- leading branch: columnar_full
- decode (decode_show, FULL — even if partial/damaged):
TOTESHOINTASTEITTEOSTONAETNEOIOEMEISNOTNDMSTETREONIHTDAEAOTELEORSOETEETHSNTHTTENREIOSOSETTTEOREEHNOSSTTTSOSHNHHNNSLTHSIIETSEOISEEEOSILEEAENNHEETSOEESDELTLISENNEOHSMTTTSRNHIDNTSTSOEOEEEESATTESITTTETIRONDAEISENTNIORSTITSNTNTTOSEETHETELESNOHNIMIENTOO
- signals: dict_rate=0.5758, quad=-6.2632
- attestation: accepts=false, coherence=0, language_conf=0.05, recoverability=0.0, damage=basin_wide
- route: Diagnosis strongly favored transposition because English-like monogram frequencies were preserved while quadgrams were scrambled. I created a keyed-columnar hypothesis, ran escalating MCP-only wide/full transform searches over 5,000 and 20,000 generated candidates, installed and read the best candidate, and requested independent verification. The reader rejected it as non-English with coherence 0 and basin-wide damage; both tested transposition branches were explicitly rejected and the investigation was closed unsolved under WF-7.

### fs7_bifid_probe
- investigation_id: 43ca1ea1f3c3
- verdict: unsolved
- family (your diagnosis): fractionation_transposition (Bifid/5×5 Polybius suspected)
- leading branch: fractionation_full
- decode (decode_show, FULL — even if partial/damaged):
ETHTEOEHTETSIRIECHOHETTEEEHCESSWCESESRAEATEHTLRHSSEATARISHSWOOIIAEURHTEIEONENLRCHSRSEESCSUTAEATHAERARTSHORSISNTCUTELEHSWLETHTEHUSTAIEESIOEOAETEAETCSTSCHSTRCTHRR
- signals: dict_rate=0.5263, quad=-6.2524
- attestation: accepts=false, coherence=0, language_conf=0.05, recoverability=0.0, damage=basin_wide
- route: Diagnosed the 25-symbol, no-boundary text and created an explicit Bifid/fractionation hypothesis plus a monoalphabetic control. The targeted `cipher_system=bifid` MCP experiment was redirected by the host router to composite substitution-transposition peeling, so I additionally ran the broadest exposed fractionation/transposition search. Its leading branch improved internal scores but remained repetitive ETAOIN-like pseudo-English; a recorded comparison rejected it as a solution, an independent reader found coherence 0 with basin-wide unrecoverable damage, and WF-7 was closed explicitly with `meta_declare_unsolved`.

### fs8_homo_transp_open
- investigation_id: a55d62a33310
- verdict: unsolved
- family (your diagnosis): transposition_homophonic
- leading branch: homo_solved
- decode (decode_show, FULL — even if partial/damaged):
SERIETRICEISTATTHROSEETHERSHESTEHTTOSSICTETHIMSIETOMERSSIAADDRETEISPARTRESSASTHALLEEIEASAPRSLIFORERMEHEEEVEIHSSSSEEGOSITDOINEHENCESEOEDDISANALITTLESSSTRAGEH
- signals: dict_rate=0.6552, quad=-5.6514
- attestation: accepts=false, coherence=1, language_conf=0.2, recoverability=0.05, damage=basin_wide
- route: Diagnosed the 156-token, 28-symbol, no-boundary stream and tested explicit homophonic and transposition-homophonic hypotheses. A full-budget Zenith homophonic baseline produced the leading partial; a full broad transform-aware homophonic search produced a weaker candidate, and the dedicated opposite-order composite cross-check failed because its bijective route lacked enough plaintext IDs. Both installed finalists were read and independently rejected as basin-wide incoherent English. I recorded the baseline as best partial without accepting it as a solution and closed the investigation explicitly under WF-7 as unsolved.
