# Repair Mechanism Rethink — Design Note and Experiment Proposal

Status: design review, **revision 3**, 2026-07-16. Author: independent
second-opinion review (Fable), verified read-only against the code as of
`4b85e20`. No code was changed. Revision 2 folds in the adjudicated results of
GPT-5.6-Sol's design review of revision 1 (ten comments; adjudication table in
§4.3) and aligns with the amended M5.3 spec
(`docs/specs/agent_v3_m5_3_control_reliability_spec.md`; the reading-budget/
usability clarifications landed at `4b85e20`, and as of `d4cbe55` the full
Part A/B amendment set is integrated normatively into the spec body). Codex's
M5.3 implementation is in flight concurrently; the recommendations below are
written to be foldable into that work, not to preempt it. The consolidated
M5.3 amendment block lives in the companion document
`docs/repair_reframe_m53_comments.md`.

### Revision 3 decisions (Codex critical review)

Revision 3 makes the following changes explicit so they are easy to audit in
the sections below:

1. recovered-reading attestation binds to a composite interpretation digest,
   not only the unchanged key-derived content hash;
2. the isolated-occurrence signature is treated as triage evidence, never as
   proof that an LLM replacement is a true transcription scar;
3. model-facing repair references use granular host-owned word/token anchors,
   not the existing coarse 120-token window id alone;
4. solver-supported null masks remain canonical structural state and are not
   folded into the noncanonical editorial-override ledger;
5. Phase 0 tests a reference implementation of the proposed deterministic
   compiler, rather than treating the current menu-backed word probe as that
   compiler;
6. the future packet producer defaults to one structured provider response,
   with at most one batched lookup round-trip, rather than inheriting the
   legacy reading episode's 16-call safety envelope;
7. residual composition admits ambiguous/unclassified cases and includes a
   human-reviewed audit sample;
8. the false-override gate uses per-claim and affected-token denominators with
   confidence bounds;
9. process-versus-evidence failure reasons are split before saturation policy
   consumes them; and
10. reserved future operation fields are rejected until implemented rather
    than accepted and silently ignored.

---

## 1. Problem framing

The user's complaint, verbatim:

> "We have a nearly complete decrypt and a smart LLM that can propose a
> high-quality corrected version, and yet we have a very difficult time
> getting it to input that data back into the plaintext, and our tools simply
> eat a lot of turns. I'm wondering if we are fundamentally doing something
> wrong and we should rethink the whole mechanism."

Four inputs now exist:

- **Position A (orchestrator, Fable main session)**: the key bottleneck is
  irreducible for genuine key errors, but the interface is wrong — repair
  should be driven by direct **span→word binding** (assertions) rather than
  prose alignment, and "recoverable reading, not key-complete" should be a
  first-class typed end state.
- **Position B (GPT-5.6-Sol, first round, as relayed)**: agrees with the
  diagnosis and a five-category error taxonomy; "we have conflated
  understanding the plaintext with modifying the cryptographic key"; proposes
  a **more radical interface fix** — making the corrected text layer itself
  the primary editable artifact rather than routing corrections through the
  key at all.
- **Position C (this review, revision 1)**: the *core* key mechanism is sound
  and its guards are correct; the *repair interface* is genuinely mis-framed —
  prose is the mandatory carrier for all repair information and a structural
  precondition for repair itself — but the fix is a re-typing of the repair
  input and end state, not Sol's inversion and not merely M5.3's host-control
  tuning.
- **Sol's design review of revision 1** (ten comments, P0-1 … P2-10): accepts
  the Position-C frame but attacks specific mechanisms in revision 1's
  proposal — chiefly that auto-promoting rejected edits to overrides launders
  guesses, that the reader→repair episode handoff survives revision 1 intact,
  and that overrides must be kept out of ordinary scoring entirely.
  Adjudicated point by point in §4.3; the accepted points are folded into the
  revised proposal in §4.4.

### 1.1 What the code actually does today (verified)

The v3 endgame path, with file/line references:

1. **Reading**: a `reading`-kind episode (`src/investigation/episodes.py:288-315`,
   budget 4 calls / 4,096 tokens / 180 s in code — the M5.2-reduced envelope
   that the amended M5.3 spec provisionally restores to 16 / 8,192 / 300)
   receives a host-built candidate packet with stable span ids
   (`src/investigation/reading.py:339-455`, 120-token windows) and returns
   prose fragments with confidences. The contract explicitly says "You do NOT
   change the key — you only read it."
2. **Repair gate**: the lead's `repair_transaction`
   (`src/investigation/loop_v3.py:958-1181`) **hard-fails with
   `fresh_reading_required`** (lines 981-990) unless a stored Reading exists
   whose `candidate_content_hash` matches the branch's current content hash.
   *A prose reading is a structural precondition for any repair.* Duplicate
   suppression (lines 1010-1028) is keyed on (source hash, reading id) — and,
   nota bene, only over transactions with `status == "installed"`; a *failed*
   (source, reading) pair can be rerun today (M5.3 Slice 2 tightens this; see
   amendment A2 in the companion doc for the necessary process-failure
   carve-out).
3. **Compilation**: the repair worker (toolset at `episodes.py:331-351`) calls
   `hypothesis_apply_reading` (`src/investigation/actions.py:376-789`), which:
   - normalizes fragment text (`repair_text` = letters/spaces/`?` only);
   - applies **direct positional matching** when the proposed length equals
     the span length (lines 539-541), **banded alignment** when the length
     delta is within `max(2, ceil(0.02·span))` (lines 542-553), and **skips
     the fragment** beyond that (lines 554-565);
   - converts every proposed-vs-decoded **mismatch** into a vote
     `cipher_symbol → proposed_letter` (lines 577-585), resolves votes by
     majority with ties dropped (lines 652-673);
   - compiles word boundaries from the fragment's spacing (lines 678-715);
   - forks, applies `set_mapping` edits + `set_word_spans` (lines 733-739),
     scores before/after, and returns a diff preview. Unresolvable
     insertions/deletions are recorded as `holes` — strings and counts only
     (lines 586-606, 752-761); nothing downstream can act on a hole.
4. **Word probe**: `hypothesis_test_word` (`actions.py:808-1024`) is the
   direct span→word primitive. It exists and is parity-by-construction with
   the word-repair library's collateral adjudication — but it is
   **singleton-only**, **same-length-only** (`span_len != len(word)` is an
   error, lines 869-878), and **rebuilds the full repair menu on every call**
   (lines 925-935). The M5.2 smoke measured 35 calls costing 456.3 s
   cumulative. The batch form `hypothesis_test_words` exists **only in the
   M5.3 spec** (Slice 3); it is not in the code.
5. **Validation + install**: the host validates that the worker's claimed
   winner is a genuinely changed snapshot (content-hash comparison,
   `loop_v3.py:1088-1122`), installs it, and marks
   `reverification_required: True`. Note what is *not* checked yet: that the
   change *improved* anything (that is M5.3 Slice 4).

Two M5.3 spec claims were verified directly in code: episode budgets are
checked only **after** a full model-emitted batch executes
(`episodes.py:1193-1196`), and the model-facing `max_tool_calls` input can
**raise** the registered cap (`episodes.py:428-434`, `int(max_calls)` with no
clamp).

### 1.2 The sharpest mechanical finding

**Majority voting is a majority over *mismatches*, not over occurrences.**
Matching occurrences contribute no vote (`actions.py:577-585` — a vote is
recorded only when `proposed_char != decoded_char`). Consequence: take a
cipher symbol Y that is correctly mapped and occurs 10 times, and a single
transcription scar where the manuscript's true symbol X was transcribed as Y.
The proposal disagrees with the decode at that one position; the vote is 1–0
in favor of the edit `Y → letter(X)`; the "majority" installs it; all ten
occurrences of Y are now decoded wrongly. **Every scar over a symbol with no
other mismatching occurrences becomes a global key-poison edit, deterministically.**

In practice the guards catch this *after the fact* — the fork's scores drop,
the diff preview shows collateral damage, the worker is told to adjudicate,
and the verify gate would refuse the declaration. So the endgame failure mode
is usually not a corrupted final answer; it is **turn-economics**: typed
knowledge the model already had ("positions 3–8 spell APPLICARE", "that token
is probably a scribal error") is flattened into untyped prose, re-inferred by
alignment into the only vocabulary available (global key edits), discovered
to be wrong by scoring, and discarded — a full round trip of episode calls,
menu rebuilds, and verification for zero net progress. That is precisely
"our tools simply eat a lot of turns."

Note the inverse of this pattern for later (§4.4, C3): "the implied global
edit fails collateral adjudication *while every other occurrence of the
symbol agrees with the current decode*" is a deterministic, host-computable
signature of an **occurrence-specific conflict rather than a global key
error**. It does *not* prove that the model's local replacement is correct, or
that the source contains a transcription scar; a wrong but attractive word
hypothesis produces the same signature. The poison bug and this useful triage
signal are the same fact read in two directions, but independent source or
review evidence is still required before the conflict may be described as a
supported scar.

---

## 2. Error-type taxonomy and expressibility

Unifying the orchestrator's four classes and Sol's five (they are the same
partition at different granularity):

| # | Error class | Scope | Correct fix | Channel today | Expressible? |
|---|---|---|---|---|---|
| E1 | Genuine key error (symbol consistently mis-mapped) | global, per symbol | key edit, applied everywhere, consistency-checked | `hypothesis_apply_reading` votes; `hypothesis_test_word` edits; `act_set_mapping` (v2) | **Yes** — the only class the vocabulary expresses well |
| E2 | Boundary / segmentation error | positional | word-span edit | `set_word_spans`; boundary compilation inside `apply_reading`; boundaries-only application supported (`character_preserving`) | **Yes, but** discovery+application are routed through the prose reading in v3 |
| E3 | Transcription scar (manuscript/OCR error: wrong symbol at one position) | positional, per occurrence | positional override; **no key edit is correct** | none | **No** — gets mis-expressed as E1 (§1.2) or lost as a `hole` string |
| E4 | Structural tokens: nulls, abbreviation marks, logograms, lacunae | per symbol *or* per occurrence | symbol-class annotation (null mask) or positional annotation | homophonic **null masks** exist (`null_mask_selected` / `null_mask_finalist` branch metadata, `investigation/candidates.py:14-19`) — global, per-symbol, untyped, installed only by the automated search path; **not writable from the repair path** | **Partially** — and not from repair |
| E5 | Editorial reconstruction (text the editor supplies that is *not in the ciphertext*: `cur[a ap]plicare`, expansions of abbreviations) | annotation layer | cannot be "applied" to anything; must be *recorded* | none (alignment sees an insertion → `hole` string, or skips the fragment) | **No** |

Verified absences: grep for lacuna/abbreviation/scar override vocabulary finds
only a comment in `analysis/homophonic_nulls.py`, a
`readable_with_local_scars` *label* in `analysis/finalist_validation.py`, and
a prompt note — no mechanism. There is no positional override of any kind,
typed or otherwise.

Two corrections to the orchestrator's point 2 ("the tooling jams (b)–(d)
through the key channel"):

- **E2 is not jammed through the key.** Word spans are first-class non-key
  branch state with their own setter, and `hypothesis_apply_reading`
  explicitly supports boundaries-only application. What E2 *is* jammed
  through is the reading→alignment pipeline.
- **E4 has a partial non-key channel** (null masks) — but it is global,
  untyped, and only reachable from the automated homophonic search, never
  from a reading or repair episode.

The accurate statement of the smell is therefore: **all repair information is
jammed through the prose-reading channel, and all content corrections through
global key edits; positional and typed corrections have no channel at all.**
The banded alignment engages exactly when the proposal's length differs from
the span — i.e., precisely on E3/E4/E5 material, the classes the key cannot
express. The mechanism guesses hardest exactly where it is least entitled to.

**Caveat (Sol P2-10, accepted): the *composition* of the endgame residual is
assumed, not measured.** Revision 1 argued as if the residual ~9% on a
borg_0109v-class page is predominantly E3/E4/E5 material. That is plausible
given the diplomatic-text examples in the M5.3 spec, but nothing in the repo
quantifies it; parts of the residual may be genuine E1 key errors, wrong-basin
content, orthographic normalization, or grading-alignment artifacts. §7 Step 0
makes measuring this composition a gating step, *before* the typed-repair
investment is sized.

---

## 3. Irreducible vs fixable

**Irreducible** (agreeing with the orchestrator's point 3, sharpened):

1. Any correction *claimed as a key fact* must apply at every occurrence of
   that symbol and survive a consistency check. This is what makes the decode
   evidence rather than hallucination; no architecture removes it. The
   borg_0077v wrong-basin risk is real.
2. The dual constraint, which any "more radical" fix must respect: **a
   positional/text-level correction channel is safe only if it is rationed,
   typed, and visible.** An unconstrained text-override layer is a
   hallucination channel — anything can be "repaired" into readability if
   overrides are free. Overrides must therefore (a) carry a type from a small
   enum, (b) be counted against an honesty budget that verification can see,
   (c) render visibly (bracketed/flagged) — and (d), strengthened per Sol
   P1-4: **never enter the canonical render or the internal scores at all.**
   Key-derived score/text, annotated readability, and override burden are
   *separate outputs*; a penalty inside a combined score is not sufficient,
   because language-model and dictionary scores would reward an invented
   correction by more than any fixed penalty deducts.
3. The ground-truth firewall: no repair, verify, scoring, or routing input
   may ever contain benchmark plaintext, keys, or alignment (M5.3 design
   principle 1). Nothing here relaxes it. (The Phase-0 experiment in §7 is an
   *oracle test* that deliberately synthesizes its proposal from ground truth
   under an isolated harness — see §7.7 for the honest statement of what that
   does and does not mean.)

**Fixable** (the actual defects):

1. **Prose as the mandatory carrier.** `repair_transaction` cannot start
   without a stored prose Reading bound to the exact content hash. The model
   is forced to encode assertions it holds in typed form ("this span is this
   word") into prose, which the host then decodes back by alignment — a lossy
   round trip through the least structured representation available.
2. **Vote-on-mismatch key inference** converts E3 scars into global key
   poison at 1–0 majorities (§1.2). There is no occurrence-weighted guard in
   the compiler itself; guards live downstream and cost turns.
3. **No vocabulary for E3/E4-positional/E5** — holes are logged strings;
   nothing consumes them.
4. **The direct primitive is hobbled**: `hypothesis_test_word` is
   same-length-only, singleton, and rebuilds its menu every call (456 s
   cumulative in one 25-turn run). M5.3 Slice 3 fixes cost, not primacy.
5. **No terminal state between solved and unsolved.** The M5 declaration gate
   is binary; the Slice 6 verifier fields (not yet landed — the current
   `_VERIFY_SCHEMA` at `episodes.py:232-245` has a single `coherence` int)
   add routing vocabulary but no *artifact* vocabulary. "Recoverable reading
   with N typed residuals" is the honest description of a solved-in-substance
   Borg page, and the system cannot say it.

---

## 4. The four inputs, adjudicated

### 4.1 Position B (Sol, first round): text-layer-primary inversion

Sol's diagnosis is correct and matches the code: reading (understanding) and
key-editing (mechanism) are conflated — most visibly in the fact that the
*reading* episode is forbidden from touching the key while being the only
sanctioned source of key edits.

The inversion — make the corrected text the primary editable artifact, derive
or check the key from it — goes too far, on the irreducibility argument in
§3. If the text layer is primary and freely editable, the key becomes
advisory; every E1 error can be "fixed" locally without paying the global
consistency cost, and readable-but-wrong basins are laundered instead of
caught. The honesty of this system lives in the fact that the rendered
candidate is *derived* from key + boundaries + (bounded) annotations. Sol's
framing is right; the full inversion sacrifices the property that makes the
decode trustworthy. It would also discard working machinery (workspace
branching, content-hash binding, collateral adjudication, verification
gating) that the M6 bake-off shows is not the bottleneck. Sol's own
second-round principle now states the same boundary better than the
inversion did: *assertions may propose key facts; annotations may preserve
interpretations; neither may silently turn an unsupported reading into
cryptographic evidence.* This note adopts that principle verbatim.

### 4.2 Position A (orchestrator): span-binding-primary + typed end state

Correct in direction on both moves, with three amendments this review adds:

1. `hypothesis_test_word` as it exists cannot be "the primary primitive":
   same-length-only means it cannot express any correction that changes
   token-to-word allocation, and singleton cost is prohibitive. The primary
   primitive is the **batched, cached assertion** — Slice 3's
   `hypothesis_test_words` plus typed headroom (§4.4, C2) — under the strict
   compilability rule of §4.4 (1 proposed char ↔ 1 ciphertext token).
2. Span binding alone does not solve E3: a span assertion over a scarred
   position implies a key edit whose collateral check fails (that is the
   *correct* outcome — the mechanism surfaces the conflict instead of
   poisoning), but without a typed annotation channel the surfaced conflict
   still has nowhere to go. Assertions and overrides are two halves of one
   fix; shipping only the first moves the dead end one step later.
3. The typed end state needs more than Slice 6's verifier fields: it needs an
   artifact representation (typed positional overrides on the branch), a
   renderer capability (a *separate* annotated reading artifact), and a
   declaration outcome distinct from solved/unsolved. Verifier vocabulary
   without artifact vocabulary changes routing, not expressibility.

### 4.3 Sol's design review of revision 1: adjudication of the ten comments

Each point was checked against the code and against revision 1's own text
before accepting. Verdicts:

| # | Sol's point | Verdict | Adjudication |
|---|---|---|---|
| P0-1 | A rejected collateral adjudication is not evidence of a scar; auto-promoting a failed edit to an override launders guesses | **Accept; revision 3 removes the overclaim** | Correct: revision 1's C3 made the rejected assertion the "natural producer" of an override. Revision 2 still overclaimed that the deterministic occurrence signature supplied independent evidence of the replacement's truth. It does not: it proves only that the conflict is local rather than a globally supported key edit. C3 now separates proposed, occurrence-conflict-supported, and source-supported annotations; only the last may be described as a supported transcription correction. |
| P0-2 | The reader→repair handoff survives revision 1; one primary reader call should return a single `InterpretationPacket`; no second LLM translates the first's interpretation | **Accept** | Correct that revision 1's separate assertion episode recreated the turn-heavy pipeline. The packet producer now defaults to one tool-free structured response, with at most one batched lookup and one final response. A later adjudicator may compare already-compiled alternatives, but no second LLM re-derives assertions from prose. See C1. |
| P1-3 | Override representation too weak; needs host-owned span, op enum, diplomatic/editorial split, scope, confidence/evidence/provenance/review | **Accept** | Revision 1's `(token_index, type, proposed_char)` cannot express E4/E5 (deletion, insertion, expansion) — precisely the length-changing material the banded aligner currently guesses at. One trim: keep the *model-emitted* core minimal (span ref, op, type, payload, confidence); the host fills provenance, evidence links, and review status. See C3. |
| P1-4 | Overrides must not participate in ordinary solver scoring; keep separate outputs; ranking stays key-derived; verifier sees both versions + ledger | **Accept, strengthened** | Revision 1 said "scoring counts them" — a penalty model. Sol is right that no penalty survives contact with a language score that rewards invented corrections. An additional architectural argument Sol did not make: the entire M5 machinery (attestation freshness, reading binding, duplicate suppression, snapshot change detection) keys on `_candidate_content_hash` over the canonical render (`decoded_text_v1`). If overrides entered that render, every override would churn content hashes and invalidate attestations and readings wholesale. Keeping the canonical render strictly key+boundary-derived preserves the hash-binding machinery *unchanged*. See C3. |
| P1-5 | Phase-0 firewall description inaccurate; it feeds true plaintext into repair input; label it an oracle compiler test, keep it out of benchmark artifacts | **Accept** | Revision 1's §7.7 claimed "no ground truth flows into any repair … input" while Arm P literally sets `repair_text` to the true plaintext. The test is legitimate — isolating the mechanism under a granted premise — but the description was wrong. Relabeled and isolated in §7. |
| P1-6 | Phase 0 cannot validate annotation recognition; add negative controls and a false-support rate; a small live Phase 1 is mandatory before adoption | **Accept, strengthened in revision 3** | A perfect oracle trivially supplies correct annotations. Arm N therefore measures false support under attractive-but-wrong proposals, with per-claim, case-level, and affected-token denominators plus confidence bounds. Phase 1 remains mandatory because Phase 0 grants exactly the interpretation quality that live runs must supply. |
| P1-7 | Stricter "span-flexible" definition: alignment-free only at exactly 1 char per ciphertext token; never compile length-changing material into key mappings; model-facing inputs use host-owned span ids, never raw token indices | **Accept; granular-anchor requirement added** | Existing host span ids identify coarse windows, not exact word/token runs. C1 now requires separate window, word, and opaque token anchors; assertions use a word id or validated contiguous token-anchor pair. Direct compilation remains legal only at exactly one proposed character per selected ciphertext token. |
| P1-8 | Landing all of M5.3 first may entrench the obsolete workflow; land orthogonal controls, amend saturation/batching around the packet now, run the oracle experiment before further prose-first investment | **Partial accept** | Agree: experiment before further prose-first investment; Slice 3 schema headroom now; saturation identity made packet-ready now. Disagree with withholding the non-orthogonal slices: Slices 2 and 4 are *interface-agnostic guardrails* — saturation and host-validated acceptance are what make any repair interface safe, and they transfer to the packet model wholesale once the identity tuple is named generically. The entrenchment risk is concentrated in three specific spots (Slice 3 hard-coding untyped same-length claims, Slice 2 keying identity on readings only, Slice 6 overloading the solved gate), each neutralized by a small amendment rather than by re-scoping a milestone that is already being implemented. See §5 and companion doc Part B. |
| P2-9 | "Reading recovered" should be a separate terminal action, not a solution-declaration subtype; excluded from solved statistics | **Accept** | A subtype on `meta_declare_solution` would flow through every consumer that branches on a solved declaration (scorer, bake-off summaries, analyzer, fallback tiering) and silently inflate solved statistics. A distinct `meta_declare_recovered_reading` gets its own gate, artifact type, and accounting bucket. It must share the hash-binding discipline of the solution gate. See C4. |
| P2-10 | Don't assume the residual 9% is predominantly transcription/null/diplomatic; measure the composition first | **Accept** | See §2 caveat and §7 Step 0. Cheap, grading-side, and it correctly sizes M5.4 before it is built. M5.1's basin-retention results are a live warning that some residual may be wrong-basin rather than inexpressible. |

### 4.4 Position C, revised: re-type the interface, keep the core

Five changes, ordered by leverage. C1/C2 replace revision 1's C1/C2/C3 input
model; C3 replaces revision 1's override channel; C4 replaces revision 1's
declaration subtype; C5 is new.

**C1 — One interpretation producer: the `InterpretationPacket`.**
The reading episode's result schema is extended so that the *same single
worker call* that produces the editorial reading also produces everything
repair needs — no second episode translates prose:

```
InterpretationPacket {
  candidate_content_hash,          # host-bound, exactly as Readings today
  editorial_reading: str,          # human-readable; may contain brackets;
                                   #   NEVER a compile source
  assertions: [                    # the ONLY compile source
    {span_ref, claim: word | boundary, payload, confidence}
  ],
  annotations: [                   # typed non-key observations;
                                   #   never compile to key mappings
    {span_ref, occurrence?, type: scar | null | abbreviation | lacuna | editorial,
     surface?, note, confidence}
  ],
  holes: [ {span_ref, note} ],     # explicit unresolved regions
  overall_confidence
}
```

- Produced by one fresh-context structured response **without tools by
  default**. If the first response explicitly requests lexical help, the host
  may perform one bounded batched corpus lookup and permit one final structured
  response. The M5.4 producer therefore permits at most two paid sends and
  does not inherit the legacy reading episode's provisional 16-call M5.3
  safety envelope.
- `candidate_content_hash` and all resolved provenance are written by the host,
  not trusted from model output.
- Model-facing references are granular host-owned anchors. The reading packet
  exposes coarse `window_id` values for context, stable `word_id` values for
  current words, and opaque `token_id` values for individual rendered tokens.
  `span_ref` is either one `word_id` or a contiguous
  `{start_token_id, end_token_id}` pair within one window. The host resolves
  and validates contiguity. Existing 120-token window ids alone are too coarse
  to bind a word assertion, and raw numeric token indices disappear from the
  model-facing surface.
- `repair_transaction` consumes the packet directly. The packet replaces the
  prose Reading as the structural precondition; the prose survives *inside*
  it as the editorial layer, feeding verification and the human reader — the
  role it was always suited to.
- Downstream of the deterministic compiler, one bounded adjudication step
  remains for surfaced conflicts and collateral failures; its input is the
  compiled evidence set, never the prose (see P0-2 adjudication).

**C2 — Deterministic compilation on the Slice-3 substrate.**
Compilation of assertions is host-side, batched, and cached (M5.3 Slice 3 is
the substrate), under the strict rule (P1-7):

- a `word` assertion compiles iff each proposed character corresponds 1:1 to
  a ciphertext token in the asserted span — alignment-free by construction.
  Crossing or redrawing word boundaries inside the span is fine (token count
  is what is preserved, not the old segmentation);
- length-changing material (abbreviation expansions, lacunae fills, editorial
  insertions) **never** compiles into key mappings — the assertion is
  rejected with the typed reason `not_expressible_as_key_edit`, pointing at
  the annotation channel;
- vote-on-mismatch inference and banded alignment are retired as key-inference
  mechanisms. Conflicting assertions touching the same symbol are surfaced as
  explicit conflicts with per-occurrence evidence, not majority-resolved;
- a `boundary` assertion changes canonical word-span state through the existing
  boundary API and therefore changes the canonical content hash;
- occurrence-level `null` observations remain annotations. A symbol-level null
  hypothesis may be promoted only through the existing solver-supported
  null-mask machinery; that produces canonical structural state, changes the
  projected token stream and content hash, and requires fresh verification. It
  is not an editorial override;
- the batch input schema carries `claim` type headroom from day one
  (companion doc, amendment B1), so M5.4 widens it without a wire break.

**C3 — Typed positional annotations, without laundering.**
A branch gains a noncanonical annotation ledger. An annotation is never a key
fact, and its support status is explicit:

1. an **explicit typed annotation** in the packet for that occurrence — the
   model's own claim, stated as a claim (P0-1). A failed assertion with no
   matching annotation records only an unresolved annotation: non-rendering,
   no readability credit, visible in the ledger as open;
2. a **host-verified signature**, deterministic and ground-truth-free: for
   an occurrence conflict, the implied global edit fails collateral
   adjudication while all other occurrences of the symbol remain consistent
   with the current key (the §1.2 inverse). This earns only
   `occurrence_conflict_supported`; it does not prove the proposed replacement
   or justify labeling the source a transcription scar;
3. **headroom in the honesty budget** (counts by type, bounded per run).

Support levels are `open`, `proposed`, `occurrence_conflict_supported`,
`independently_reviewed`, and `source_supported`. Independent linguistic review
can produce only `independently_reviewed`; it remains an editorial conjecture.
Only manuscript/OCR evidence or corroborating transcription evidence can
produce `source_supported`. A semantically attractive local replacement
without source evidence is rendered with uncertainty and never reported as a
cryptographically established transcription correction. The honesty budget
applies separately to each support level.

Representation (P1-3): model-emitted core
`{span_ref, occurrence, op ∈ {replace, delete, insert_after, expand},
type, payload (diplomatic surface vs editorial expansion as separate fields),
confidence}`; host-filled `{resolved token position, evidence links,
provenance, review_status}`. The annotation ledger is occurrence-level and
noncanonical. Symbol-level null masks remain canonical structural solver state
under C2 and use a separate structural-hypothesis record.

Scoring and rendering (P1-4, strengthened): annotations **never** enter the
canonical render or any internal score. Three separate outputs:

- the **key-derived text and scores** — canonical, hash-bound; every existing
  consumer (attestation binding, duplicate suppression, ranking, fallback
  selection) is untouched;
- the **annotated reading** — a separate artifact with its own hash, where
  conjectural or supported additions render visibly (`applic[a]re`, with
  support status available to the presentation layer);
- the **annotation ledger** — burden metrics: counts by type and support level,
  affected-token fraction, and open vs reviewed.

Native branch ranking stays key-derived. The verifier receives both texts and
the full ledger only in recovered-reading verification; ordinary solution
verification receives key-derived text alone.

**C4 — `meta_declare_recovered_reading`: a separate terminal action** (P2-9).
Not a subtype of `meta_declare_solution`. It uses a distinct recovered-reading
attestation whose binding digest covers
`{key_content_hash, interpretation_packet_hash, annotated_reading_hash,
annotation_ledger_hash}`. Any change to the packet, annotation text, support
status, or ledger makes this attestation stale. The recovered-reading verifier
sees the key-derived text, visibly annotated reading, and complete burden
ledger; the ordinary solution verifier sees only key-derived text and remains
bound only to its canonical content hash.

The recovered-reading gate additionally requires high
`semantic_recoverability` (Slice 6 field), explicit uncertainty, and annotation
burden within budget. Its artifact carries both hashes, the annotation census,
and verifier fields. It is excluded from solved-result statistics everywhere
(scorer, bake-off summaries, analyzer); it exists precisely so that a
solved-in-substance diplomatic page stops masquerading as either "solved" or
"unsolved".

**C5 — Measure before building** (P2-10). The residual-composition study of
§7 Step 0 runs before M5.4 is scoped. If genuine key errors dominate the
residual, the annotation channel loses urgency and search/repair quality is the
real frontier; if E3/E4/E5 dominates, the expressibility argument of §2 is
confirmed with numbers.

---

## 5. Relation to Codex's in-flight M5.3

M5.3 is **necessary and correctly aimed, but not sufficient** for this
problem. The consolidated amendment block — the eight reliability
clarifications from the spec review plus the design-seam amendments — lives in
`docs/repair_reframe_m53_comments.md`; the table below is the summary view.
On Sol's P1-8 (partially accepted, §4.3): all slices land, because Slices 2
and 4 are interface-agnostic guardrails, and the concentrated entrenchment
risks are each neutralized by a small seam amendment rather than by
re-scoping in-flight work.

| Slice | Content | Relation to this note |
|---|---|---|
| 1 (budgets) | hard per-call enforcement, no model-raised caps, cost ceiling, reading envelope + usability acceptance | Orthogonal; land as amended by `4b85e20`. Both enforcement defects were verified in code (§1.1). Cost-ceiling scope needs the mid-episode clarification (amendment A4). |
| 2 (saturation) | repair-cycle identity, `repair_exhausted` | Interface-agnostic guardrail; land it. Seams: identity component named `interpretation_id` (B2) so packets join without a state migration; process- vs evidence-failure carve-out (A2); `repair_exhausted` must be added to the episode-kind map, whose unknown-phase fallback is currently *all kinds* (A5). |
| 3 (batch/cache) | `hypothesis_test_words`, menu cache | **A performance substrate, not the proposed compiler itself.** Seams: typed word claims, granular host-owned span references, and non-same-length rejection as typed data (B1); future fields are documented but rejected until implemented; cache keyed on exact builder inputs or a proven-sufficient proxy (A8). Phase 0 supplies a separate reference compiler so it actually tests C2. |
| 4 (host-validated acceptance) | edits bound to tool evidence, acceptance policy | Interface-agnostic guardrail; land it. An assertion-driven repair produces *cleaner* evidence for it (explicit claims instead of inferred votes). Feeds the A2 failure classification. |
| 5 (typed experiments) | typed experiment config | Orthogonal; land as specced. |
| 6 (diplomatic verification) | multi-field verifier contract | **The routing half of C4.** Land the fields and routing. Two cautions: the "only `reader_accepts_as_solution` satisfies the gate" sentence silently reverses documented design C6 and needs an explicit decision plus a consumer migration map (A3); and the solved gate must not absorb "recovered reading" semantics — that is C4's separate terminal, deferred to M5.4 (B3). |
| 7 (observability) | telemetry, analyzer parity | Orthogonal; land it. Claim the pre-existing failing `test_lead_context` assertion (A6); the committed analyzer fixture must be trimmed of ground truth and message bodies (A7). |

Recommendation unchanged in substance from revision 1: M5.3 lands in full
with the amendments; C1–C4 become the follow-on milestone (working name
**M5.4 "typed repair"**), whose go/no-go is decided by §7 — now three gates:
the residual-composition measurement (Step 0), the oracle compiler test with
negative controls (Phase 0), and a small mandatory live phase (Phase 1).

The one thing *not* to do is treat M5.3's host-control work as the answer to
the user's complaint. M5.3 makes the prose-first loop bounded, deduplicated,
and cheap; it does not change what the loop can express. If the diagnosis in
§2 survives Step 0's measurement, a perfectly tuned prose-alignment loop
still cannot close an endgame whose residual is typed E3/E4/E5 material — it
will just fail with excellent budget discipline.

---

## 6. Verdict

**The user's instinct is substantially correct, and neither "throw it all
out" nor "M5.3 will fix it" is the right response.**

- The core mechanism — key + boundaries as the honest derived artifact,
  branch isolation, content-hash binding, collateral adjudication,
  verification-gated declaration — is *sound* and should not be inverted
  (contra the radical reading of Position B).
- The repair *interface* is *fundamentally mis-framed*, in the strict sense
  that no amount of tuning fixes it: prose is the structurally mandatory
  carrier of repair information (`fresh_reading_required`), the compiler
  re-infers what the model already knew in typed form, the inference rule
  converts transcription scars into global key edits at 1–0 "majorities",
  and three of the five error classes in the endgame residual have no
  vocabulary at all. These are architecture facts, verified in code, not
  prompt or budget problems.
- The fix is bounded, and after Sol's review it is also *tighter* than
  revision 1's version: one interpretation producer emitting a typed packet
  (no reader→repair translation), a deterministic 1-char-per-token compiler
  (no alignment guessing), a typed annotation channel whose support level is
  explicit and never inferred from semantic plausibility alone, strictly
  separated outputs (no score
  contamination, no hash churn), and a separate honest terminal state (no
  solved-statistics pollution). Keep everything else.

Where this disagrees with the orchestrator: point 2 overstates the key
bottleneck (boundaries and null masks are existing non-key channels — the
real chokepoint is the prose-reading channel and the absence of positional
vocabulary); point 5 understates what "primary" requires (batch +
span-flexible + typed, and span binding without the annotation channel just
moves the dead end). Where this disagrees with Sol: the first-round inversion
trades away the consistency property that makes the whole exercise
cryptanalysis rather than creative writing; on the second round, only two
points are resisted, both narrowings rather than rejections — bounded
adjudication over compiled evidence survives P0-2, and M5.3 lands whole
rather than partially under P1-8 (§4.3). Where this disagrees with "just
land M5.3": §5, last paragraph.

---

## 7. Experiment program: does the mechanism, not the model, eat the turns?

### 7.0 Step 0 — residual-composition measurement (grading-side; runs first)

Before any mechanism experiment, measure what the endgame residual *is*
(P2-10). Using stored artifacts (the M5.1/M5.2 borg_0109v and borg_0045v
candidates at 83–96% char accuracy) and benchmark ground truth **on the
grading side only** — the same firewall class as `benchmark/scorer.py` —
classify every residual character/word error into:

- **E1**: symbol consistently mis-mapped (all occurrences wrong the same way);
- **E3 candidate**: isolated occurrence conflict (symbol correct elsewhere;
  single-occurrence mismatch). This is not labeled a transcription scar
  without corroborating source evidence;
- **E4/E5**: editorial material — bracketed insertions, lacunae,
  abbreviation expansions in the diplomatic plaintext with no ciphertext
  counterpart;
- **E2**: boundary/segmentation mismatches;
- **grading artifacts**: alignment/normalization effects of the scorer itself;
- **ambiguous/unclassified**: the available transcription, alignment, and
  plaintext do not justify a unique causal label.

Each classification carries a confidence and the evidence used. At least 10%
of labeled residuals, every low-confidence label, and every proposed E3 label
receive human review against the canonical transcription and available source
notes/images. Report both raw automated fractions and reviewed fractions; do
not silently redistribute `ambiguous/unclassified` cases.

Output: per-page fractions by class and confidence. This report sizes M5.4: a residual
dominated by E1 says the frontier is search/repair quality, not
expressibility; a residual dominated by E3/E4/E5 confirms §2 with numbers.
No new runtime code; one grading-side script over stored artifacts. It never
touches routing, repair, scoring, or verification.

### 7.1 Question and design principle (Phase 0)

Adjudicate between two hypotheses with the LLM held out of the loop:

- **H-mech** ("fundamentally doing it wrong"): given a *perfect* corrected
  text, the prose-alignment path cannot reliably close a small typed error
  set — it poisons keys on scars, skips or mangles length-changing material,
  and cannot reach the expressibility ceiling — while direct span-binding
  closes the key-error subset cleanly and surfaces the rest.
- **H-tune** ("sound but inefficient"): given the same perfect input, both
  paths close what is closable; the difference is calls and latency, and
  M5.3 Slices 1–3 are the complete answer.

The user's premise is "a smart LLM that can propose a high-quality corrected
version." Phase 0 *grants that premise mechanically*: the mechanism arms
receive the same synthetic, perfect proposal, and we measure only what the
apply mechanism does with it.

**What this makes Phase 0 (P1-5, accepted): an *oracle compiler test*, not a
benchmark run and not a ground-truth-free evaluation.** The oracle proposal
is derived from ground truth and is deliberately fed into the repair
mechanism's production input — that is the entire point of an isolated
mechanism test under a granted premise. It is therefore:

- run outside normal benchmark execution, by a dedicated script;
- written to a dedicated, clearly named artifact area
  (`artifacts/oracle_compiler_test/`), with every artifact carrying an
  explicit `"oracle_compiler_test": true` marker;
- excluded from benchmark summaries, loaders, and any results table that
  reports solver capability;
- never described as ground-truth-free. (Revision 1's §7.7 claimed no ground
  truth flowed into repair inputs; that was inaccurate and is corrected in
  §7.7 below.)

### 7.2 Generation (uses landed code only)

Monoalphabetic substitution first (the endgame in question); the new family
generator's polyalphabetic/transposition families are out of scope here.

- **Base cases**: `src/testgen/builder.py::build_test_case` with
  `TestSpec(language, approx_length=200, word_boundaries=True, seed=s)` for
  the boundary regime (G1) and `word_boundaries=False, approx_length=250`
  (G2). Languages `en` and `la` (both have dictionaries). 20 seeds per cell.
- **Typed damage injection** (harness-side, recorded in a sidecar ledger):
  - **N key errors** (E1): derange N mappings among symbols with ≥3
    occurrences. N ∈ {2, 4}.
  - **M transcription scars** (E3): after encipherment, replace the
    ciphertext token at M random positions with a different alphabet symbol.
    Stratify so at least one scar lands on a high-frequency symbol and one on
    a ≤2-occurrence symbol. M ∈ {0, 3, 6}.
  - **K boundary errors** (E2, G1 only): delete or shift K word-boundary
    cuts. K ∈ {0, 3}.
- **Start state**: a `Workspace` whose `main` branch carries
  `true_key ∘ derangement` and the corrupted spans — a ~88–96% decode with a
  *known, typed* residual, mimicking the real endgame.
- **Expressibility ceiling**: decode the *scarred* ciphertext with the *true*
  key and *true* boundaries, score with `benchmark/scorer.py::score_decryption`
  → `ceiling_char_acc` (< 100% whenever M > 0). This is the honest maximum
  any key+boundary repair can reach; reaching it and *stopping* is the
  success behavior.

### 7.3 Arms

Arm P and the optional S-menu comparator drive landed composites through
`WorkspaceToolExecutor` + `execute_composite` in-process. Arm S-ref and S+A use
the explicitly specified reference compiler in the isolated harness so the
experiment can test the proposed mechanism before it ships.

- **Arm P — prose alignment (today's path).** Build reading fragments exactly
  as a reading episode would: 120-token windows (matching
  `build_candidate_reading_packet`), `repair_text` = the true plaintext for
  the window with true spacing, confidence 1.0, host token provenance. Apply
  via one `hypothesis_apply_reading` call (then one repair round-trip per
  residual, if any).
- **Arm S-ref — proposed deterministic span compiler.** The experiment script
  contains a small, pure reference implementation of C2: resolve host-owned
  granular anchors, require exactly one proposed character per ciphertext
  token, emit exact symbol→letter assertions, surface conflicts without
  majority voting, and compile boundaries separately. It performs no menu
  generation, language scoring, or automatic install decision. Assertions that
  compile under these rules are applied together on one fork, followed by the
  same collateral measurements used for all arms (in Arm N this includes
  adversarial assertions that compile; nothing is pre-filtered for truth).
  This is the arm that tests the reframe.
- **Arm S-menu — current direct word-probe substrate (secondary comparator).**
  Run the same oracle word assertions through `hypothesis_test_words` after
  Slice 3 lands. This arm measures parity and menu/cache economics; it is not
  treated as the proposed deterministic compiler. Where the
  1-char-per-token rule blocks a correction, record `inexpressible_by_arm`.
- **Arm S+A — assertions + typed annotations (the C3 end state, simulated).**
  As Arm S-ref, but the oracle also emits typed annotations for injected
  occurrence errors. The harness assigns support levels exactly as C3 does:
  explicit claim plus deterministic signature can reach only
  `occurrence_conflict_supported`; source support is available only because the
  oracle ledger supplies source evidence, and is labeled as oracle-only in the
  artifact. A rejected assertion with no matching annotation remains open.
  Final artifact = key + boundaries + annotation ledger; annotated text is
  scored separately from key-derived text.
- **Arm N — negative controls (P1-6).** As Arms S-ref and S+A, but the "oracle"
  is adversarial: proposals are incorrect yet linguistically attractive —
  same-length dictionary words that differ from the truth, plausible
  length-changing variants (abbreviation-like contractions/expansions), and
  scar *claims* on positions that are not scarred. Measures whether the
  support rules reject attractive falsehoods. Include isolated wrong-word
  proposals, locally fluent wrong-basin passages, and false annotation claims.
  Run at the same M/N/K matrix.

### 7.4 Metrics (per case × arm)

1. `char_acc_final` vs `ceiling_char_acc` and vs start — for the key-derived
   text; the override-rendered text is reported separately and never enters
   the arm's primary accuracy.
2. `key_error_closure`: fraction of the N injected E1 errors corrected.
3. `poison_count`: installed key edits touching symbols whose mapping was
   already true (predicted for Arm P at M>0 by §1.2; the per-scar prediction
   is deterministic and checkable).
4. `occurrence_error_handling`: silently-poisoned / surfaced-and-rejected /
   open annotation / occurrence-conflict-supported / source-supported.
5. `boundary_closure` (G1).
6. `mechanism_calls` and `wall_seconds` (menu-rebuild economics; ties into
   Slice 3's ≥70% target).
7. annotation precision/recall by support level against the injected ledger
   (Arms S+A and N).
8. **False-support metrics** (Arm N), with explicit denominators:
   - false supported annotations / all false annotation claims;
   - cases containing at least one false supported annotation / all cases;
   - falsely affected tokens / all rendered tokens; and
   - 95% Wilson confidence intervals for the per-claim rate.

### 7.5 Pre-registered adjudication

- **H-mech is supported** if, at M ≥ 3 with a perfect proposal, Arm P has
  median `poison_count ≥ 1` or median `char_acc_final < ceiling − 1 pt`,
  while Arm S-ref achieves `key_error_closure ≥ 0.9` with strictly less poison at
  comparable call counts. → proceed toward M5.4, subject to the annotation
  gate below and Phase 1.
- **H-tune is supported** if Arm P reaches the ceiling with ~zero poison and
  the arms differ only in calls/latency. → M5.3 Slices 1–3 are the complete
  answer; C1/C3 are dropped; C4 may still be justified on declaration-honesty
  grounds alone but loses its urgency.
- **Both fail** similarly → the bottleneck is elsewhere (proposal quality or
  verification), and neither reframe is justified by this evidence; revisit
  after Phase 1.
- **Annotation-channel gate** (independent of the above): C3 is adopted only if
  Arm S+A shows high recall for occurrence conflicts **and** Arm N produces no
  false `source_supported` or `independently_reviewed` annotations, a per-claim
  false `occurrence_conflict_supported` rate below 1% whose 95% Wilson upper
  bound is below 2%, and a falsely affected-token fraction below 0.1%. Report
  case-level failures separately; one high-impact false annotation blocks
  adoption even when aggregate rates pass. If the support checks cannot reject Arm N's
  attractive falsehoods, C3 must be reworked or dropped. If S+A ≈ S-ref on
  accuracy and the ledger adds nothing, the Slice 6 verifier vocabulary
  suffices and no artifact-level override layer is needed.

### 7.6 Phases and cost

- **Step 0 (first)**: the residual-composition report (§7.0). Grading-side
  only; can be prepared as a standalone script without touching `src/`.
- **Phase 0 (the oracle compiler test)**: one new script,
  `scripts/run_repair_mechanism_experiment.py` (~350 lines: build → corrupt →
  reference compiler → optional production composites → score → JSONL
  summary), plus the Arm N adversarial proposal generator. Full matrix runs
  offline for $0. Arm P and Arm S-ref do **not** wait for Slice 3; run them
  after Step 0 so the architectural decision is not circular. Add Arm S-menu
  after Slice 3 lands to measure production parity and cache economics.
- **Phase 1 (mandatory, small, before any M5.4 adoption — P1-6)**: even if
  Phase 0 is decisive, a small live phase runs before the packet workflow is
  adopted, because Phase 0 grants exactly the premise (perfect interpretation
  quality) that live runs must supply — packet-production quality is
  unmeasurable offline. Same damaged branches, real episodes: a reading
  episode feeding today's `repair_transaction` (Arm P-live) vs a
  packet producer capped at two sends feeding the deterministic compiler
  (Arm S-live), `gpt-5.5`, measuring end-to-end turns/$ to ceiling, poison
  rates, packet usability (assertion validity rate, annotation quality), and
  false-override behavior under the M5.3 harness. A handful of runs with
  separate spend approval. Phase 0 decides *whether* M5.4 is designed;
  Phase 1 decides *whether it ships*.

### 7.7 Firewall (corrected per P1-5)

- **Step 0** uses ground truth on the grading side only — identical in kind
  to the existing scorer. Its outputs are reports, never runtime inputs.
- **Phase 0 is an oracle test and says so.** The harness uses ground truth to
  (a) construct the damaged start state, (b) synthesize the oracle and
  adversarial proposals — which *are* ground-truth-derived and *are* fed into
  the repair mechanism's production input, deliberately — and (c) grade
  post-hoc. The honest claim is not "no ground truth flows in"; it is: the
  mechanisms under test treat the proposal as an untrusted input exactly as
  in production, the test is isolated from benchmark execution and artifacts
  (dedicated script, dedicated marked artifact dir, excluded from all
  capability summaries), and nothing from this harness ships into runtime
  code paths.
- **Phase 1** runs under the unmodified production firewall: workers receive
  only the candidate packet; no benchmark plaintext, key, alignment, or
  accuracy enters routing, repair, scoring, retries, verification, or
  declaration (M5.3 design principle 1 preserved verbatim).
- **The annotation channel (C3) is itself firewall-relevant**: annotations are
  typed proposals, never ground truth. Deterministic occurrence-conflict
  support is explicitly weaker than source support, and Arm N's false-support
  gate exists so the channel cannot silently substitute assertion for
  evidence. Annotations never alter canonical key-derived text, its hash, or
  any internal score, so no consumer of the honest artifact can be influenced
  by them unknowingly.
