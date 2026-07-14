# Spec: Agent Loop v3 — Milestone M5 (Verification-Gated Declaration)

Parent design: `docs/specs/agent_v3_design.md` (C6 + the M5 milestone;
amendments A2 declaration-policy injection, A6 attestation binding, A9
failure semantics — all binding). Spec author: Fable. Implementer:
coding sub-agent. This spec receives its own Fable review before
implementation.

## Why (the problem M5 fixes)

Every v3 acceptance run (M1/M2/M3) showed the same pair of failures the
lean gate-free brief creates: the lead either runs its full turn budget
without declaring (→ `fallback_declared`), or — the older, worse v2
failure — declares a confident, readable, WRONG basin (borg_0077v:
dict_rate high, text reads, plaintext wrong). Self-attestation cannot
catch the second: a context that spent 20 turns producing a text is
biased to believe it. M5 adds an INDEPENDENT reader — a fresh-context
`verify` episode that sees only the candidate plaintext — as both the
noise filter (blocks/flags bad declarations) AND the positive signal
(gives the lead the coherence confidence to declare a good one).

Scope: the `verify` episode kind, the attestation record + binding, the
attestation-check `DeclarationPolicy` for v3, firewall extension. Builds
on M2 episodes + M3's composite/reading surfaces. v2 untouched. No new
coercion — the gate is one structural check, not the v2 cascade.

## Part 1 — The `verify` episode kind

- New episode kind `verify` (registered in `EPISODE_KINDS`, session-
  factory routing per A7). Its EpisodeSpec:
  - `inputs`: the candidate plaintext string + `language` ONLY. NOT the
    cipher, NOT the key, NOT any score, NOT the lead transcript, NOT the
    branch metadata. This is the most leak-sensitive surface in the
    program — the whole value is independence.
  - `toolset`: EMPTY (text-only judgement; no observe/score/search
    tools). The worker reads and judges.
  - `result_schema` (`_VERIFY_SCHEMA`): `{coherence: int 0-10,
    reader_accepts: bool ("would a fluent reader of <language> accept
    this as real, if damaged, text?"), gloss: str (clause-level
    translation/paraphrase, ≤400 chars), anomalies: list[str]
    (non-words, broken syntax spans, wrong-language runs), confidence:
    "high"|"medium"|"low"}`.
  - system prompt = the verify contract: "You are a fluent reader of
    <language>. Here is a candidate decipherment. Judge ONLY whether it
    reads as real (possibly damaged/partial) <language>: gloss what it
    says, list anomalies, and state whether a fluent reader would accept
    it. You have no other information and need none." No cryptanalysis
    framing (avoids the refusal the INV experiment hit), no scores to
    anchor on.
  - budget: 1 send, small max_tokens (this is a single judgement, not a
    loop); `verify` needs no tool loop — it is a one-shot structured
    read. If the episode runtime assumes ≥1 tool call, allow a
    zero-tool-call terminal `episode_submit_result` path (verify submits
    its result as its only action).
  - default model tier: cheap is fine (the INV smoke showed luna-class
    reads adequately for coherence; make it config, default the lead's
    provider's cheap tier or the lead model — decide and state).

## Part 2 — Attestation record + binding (A6)

- `AttestationRecord` in `InvestigationState`: `{branch, content_hash,
  renderer_id, episode_id, coherence, reader_accepts, gloss, anomalies,
  created_turn}`. Stored in state (serialized; survives resume).
- **Content hash (A6, exact):** `content_hash = sha256(candidate_string)`
  where `candidate_string` is produced by ONE named renderer
  (`renderer_id`, e.g. `"decoded_text_v1"`) — the exact string sent to
  the verify episode. Declaration recomputes the hash with the SAME
  renderer over the branch's current decoded text and matches. There are
  two decode representations in the codebase (key-derived `apply_key`
  vs metadata `decoded_text`); the renderer must pick deterministically
  and identically at attest-time and declare-time — pin which, and pin
  it in a test (this is the F7-class hash-mismatch trap from the design
  review).
- A `verify` episode's result is turned into an AttestationRecord by the
  lead (workers write nothing to state, per A1) keyed to the branch whose
  text was verified.

## Part 3 — Attestation-gated declaration (A2)

- Extend the injected `DeclarationPolicy` set: v2 injects `V2GatePolicy`
  (unchanged), M1-M4 v3 injects `NoGatesPolicy`, M5 v3 injects
  `AttestationPolicy`.
- `AttestationPolicy.check(executor, args)`: `meta_declare_solution` on
  branch B is allowed iff an AttestationRecord exists for B whose
  `content_hash` matches B's current rendered text (same renderer). If
  none exists or the hash is stale (text changed since attestation),
  return a structured block: `{reason:
  "attestation_required"|"attestation_stale", branch, how: "run a verify
  episode on this branch, then declare"}`. This is ONE check, not a
  cascade.
- **Weak attestation does NOT hard-block** (design C6): if an
  attestation exists but `reader_accepts` is false / coherence low, the
  lead MAY still declare (a deliberate partial/unsolved-leaning call) —
  the block only fires on ABSENT or STALE attestation. The declaration
  carries the attestation (coherence/anomalies) into the artifact so a
  weak-but-declared solve is visibly weak. `meta_declare_unsolved` is
  NOT attestation-gated.
- Phase 0 `fallback_declared` semantics preserved: exhaustion/error auto-
  declare still sets `fallback_declared` + `auto_declared`, and does NOT
  require an attestation (it's a give-up, not a claim).
- Iteration mapping + the policy-object plumbing reuse M2's A2 machinery
  (the policy is already injectable; M5 adds the third implementation).

## Part 4 — Lead brief + observability

- One brief line (context.py): "Before declaring a solution, run a
  `verify` episode on your best branch — a fresh reader will tell you if
  it truly reads as <language>. Declaration requires it." This is the
  positive-signal half (fixes run-to-fallback: the lead now has a cheap
  action that yields the confidence to declare) — NOT a nag; it names the
  one required step.
- Artifact: `attestations` list (additive); the declaration records its
  attestation. `inspect_artifact.py` renders the attestation (coherence,
  reader_accepts, anomalies) on the declared branch.

## Part 5 — Firewall (the highest-stakes property)

- `verify` episode inputs are text-only by construction; extend
  `assert_no_ground_truth_leak` coverage to the verify episode's rendered
  input (candidate + language) — no cipher, no key, no ground-truth
  plaintext (the candidate is the SOLVER's output, which is allowed; but
  assert the true benchmark plaintext, when different, never reaches the
  verify prompt via any metadata channel).
- The verify worker gets a fresh executor with an EMPTY toolset and NO
  workspace access beyond the candidate string — verify the A1 isolation
  holds (no lead-state bleed into the verify context; reuse the M2
  no-bleed test pattern).

## Part 6 — Tests

- `verify` episode: schema validation, the zero-tool-call submit path,
  the no-bleed/firewall assertions, cheap-model routing.
- Attestation binding: attest→declare hash match (same renderer both
  ends); a STALE case (attest, then mutate the branch key, then declare →
  `attestation_stale` block); the key-vs-metadata renderer determinism
  pin.
- `AttestationPolicy`: absent→blocked, present+match→allowed, weak-but-
  present→allowed-with-recorded-weakness; v2 declare-gate tests still
  green (V2GatePolicy unchanged); NoGatesPolicy (M1-M4) unaffected.
- **The wrong-basin fixture (the design's M5 acceptance):** a scripted
  readable-but-wrong candidate (borg_0077v-class — high dict_rate,
  reads, wrong plaintext). The verify episode's `reader_accepts` /
  `anomalies` must flag it (low coherence or anomalies present) where v2
  self-attestation declared clean. Since the verify verdict comes from a
  model, use a fake verify-session scripted to return the realistic
  "reads as words but not coherent sentences" verdict, and assert the
  attestation records the weakness and (if the lead declares anyway) the
  declaration carries it. A real-model version is the acceptance run.
- fallback_declared path still needs no attestation.

## Acceptance (compute, report)

1. Suite green (baseline: record — main is ~1232 + INV-0 if it lands
   first; the v2 A2 extraction outcome unchanged).
2. Real-model check (<$3): a v3 run on a page where the agent reaches a
   real reading — ideally one Borg page WITH `--no-automated-preflight`
   so the agent actually solves and then must verify (the M1-M3 lesson:
   preflight-solved pages never exercise the new machinery). Report:
   did the lead run a verify episode, the attestation's coherence/
   reader_accepts, whether declaration was gated correctly, and whether
   this changed the run's outcome vs an M4 run of the same page (did it
   avoid a bad declaration / did it declare where M4 ran to fallback).
3. The wrong-basin real check if cheap: point a verify episode at a
   known readable-but-wrong decode (the stored borg_0077v artifact's
   text) and report the real reader's verdict — does it catch it.

## Out of scope

Deletion of v2 self-attestation (`meta_attest_reading_comprehensibility`
retires with v2 at M6); the M6 bake-off; multi-branch verify
orchestration (one branch at declare time is enough for M5).

## Deliverables

Files changed, suite counts, the acceptance results (verify-episode
usage, attestation verdicts, gate behavior, wrong-basin catch), the
renderer-determinism confirmation, deviations. No commits.

## Post-review amendments (BINDING — Fable review: READY WITH AMENDMENTS)

**F1 (A6 renderer — the correctness core).** The renderer is
`_decoded_text_for_panel` (`src/agent/loop_shared.py:49`), pinned as
`renderer_id="decoded_text_v1"`; BOTH attest-time and declare-time import
and call that same function. Rationale (binding): it is the function that
fills `BranchSnapshot.decryption` (`loop_shared.py:197`), i.e. the exact
string the benchmark scores (`runner_v2.py:260-262`) — attesting any
other string certifies text the run is not scored on. Do NOT use
`WorkspaceToolExecutor._branch_decoded_text` (`tools_v2.py:~4433`) — it
skips word-span re-spacing and differs in whitespace on metadata-carrying
branches. Tests: renderer determinism; a NEGATIVE test on a
metadata+word_spans fixture showing `_branch_decoded_text` differs
(documents the trap). Stale-detection nuance (binding on tests): key
mutators do not invalidate `metadata["decoded_text"]`, so on a branch
carrying both, key edits don't change the rendered string and the
attestation legitimately stays fresh (attested == declared == scored).
The `attestation_stale` test MUST therefore use a KEY-rendered branch
(no metadata decoded_text); add a companion test documenting the
metadata-carrying consistent-but-frozen behavior.

**F2 (verify input path — no score leak).** `_dispatch_episode_run`
(loop_v3.py) special-cases `kind=="verify"`: render `candidate =
_decoded_text_for_panel(...)` for the named branch AT DISPATCH TIME;
compute `sha256(candidate.encode("utf-8"))` there; build the EpisodeSpec
with `inputs={"candidate_text": ..., "language": ...}` and `branches=[]`
(the empty episode workspace falls out of `_build_episode_workspace`).
`build_episode_context` renders ONLY the candidate + language framing for
this kind — no branch cards, no `dict_rate`, no `quad`, no decode
windows (scores in the verify prompt defeat its independence, and the
ground-truth leak-assert cannot catch them). On `status=="ok"` the
DISPATCHER code (not the lead model) writes the AttestationRecord with
the pre-computed hash — mirror the reading-compile precedent
(`loop_v3.py:318-329`). Test: capture the rendered verify blocks (the
`EpisodeFake.blocks_seen` pattern, `test_episodes.py:801`) and assert no
`dict_rate`/`quad`/branch-card text.

**F3 (brief edits — three, enumerated).** (a) Replace the
`context.py:103` "no gate prerequisites — declare when the reading
justifies it" sentence with the verify-then-declare line. (b) Rewrite
the preflight-solved guidance (`context.py:80-82`) to "run a `verify`
episode on it, then declare" (as written it instructs a declaration M5
would block every time). (c) Update the episode-kinds enumeration
(`context.py:84`) to include `verify` (and check it names `repair`).

**F4 (policy plumbing).** (a) AttestationPolicy gets the records via
constructor injection of the live reference —
`AttestationPolicy(attestations=state.attestations)` in run_v3, per the
repair_agenda/finalist_sessions precedent (`loop_v3.py:150-162`);
recompute the current text over `executor.workspace`. (b) Subclass
`NoGatesPolicy`, overriding ONLY `check_declare_solution` (subclassing
bare DeclarationPolicy would silently drop the v3 neutral finalize-phase
guard, `tools_v2.py:2297-2316`); do not override
`check_declare_unsolved`. (c) Block dict uses the v2 shape:
`{"status": "blocked", "accepted": False, ...reason/branch/how}` —
`loop_v3._tool_status` keys off `status`.

**F5 (verify kind registry).** `EPISODE_KINDS["verify"]`:
`toolset=frozenset()`; `EpisodeBudget(max_tool_calls=1,
max_output_tokens≈1024, wall_clock 60–90s)` (wall-clock is the real
bound; "1 send" is not a budget dimension and the nudge loop may add a
send). Language interpolation: EPISODE_KINDS contracts are static —
format the contract in `run_episode` after language resolution (or carry
language in the context text); pick and state in code comments.
`coherence: int 0-10` range is ADVISORY (the local validator has no
min/max support) — clamp at read time in the dispatcher. Update
`EPISODE_RUN_TOOL`'s prose description (the enum self-updates; the prose
does not) and TOOLS.md per CLAUDE.md.

**F6 (verify model).** Default = the lead model. Programmatic override =
run_v3's existing `episode_models["verify"]` kwarg. CLI plumbing is OUT
OF SCOPE for M5.

**F7 (acceptance provisioning).** Acceptance #2 reports gate behavior +
attestation content on the single run; the "vs an M4 run" comparison is
OPTIONAL — only if an existing artifact for the same page/config exists;
do NOT run a new M4 baseline (no spend doubling). Acceptance #3 source
text: the stored borg_0077v decode under
`artifacts/panel_borg3*/borg_single_B_borg_0077v/` or
`artifacts/baseline_borg10/borg_single_B_borg_0077v/` (or INV-0's
`tests/fixtures/borg_0077v_basin.txt` if INV-0 has landed).

**F8 (wrong-basin test framing).** The scripted-fake fixture tests
gate/record PLUMBING (absent→block, match→allow, weak→carried-weakness);
real detection is acceptance #3, not the unit test. ADD the missing gate
test: a declare attempt with NO attestation on a branch that would have
passed every v2 gate — proving the new check is the operative control.

**F9 (late-turn risk — named + mitigated).** New failure mode: a lead
that first attempts declaration on its final turn gets blocked with zero
turns left → fallback_declared (strictly worse than M4 on that path).
Mitigation (cheap, in scope): when turns remaining ≤ 2 and the current
best branch lacks a fresh attestation, the context builder appends one
hint line ("no attestation on <branch>; run verify now if you intend to
declare"). Acceptance #2 watches for this path.

**F10 (landing order + v2 note).** INV-0 lands FIRST; M5 implements in a
worktree and reconciles (collisions are shallow: disjoint tools_v2
regions, shared firewall test + investigation/__init__.py). Explicitly
OUT OF SCOPE until M6: `_tool_score_panel`'s v2 text recommending
`meta_attest_reading_comprehensibility` (v2 surface; retires with v2).

**F11 (mechanical pins).** Additive optional attestation field on
`SolutionDeclaration` + `RunArtifact.attestations` list; state
round-trip defaults for old artifacts; name the new store distinctly
from v2's `executor._reading_attestations` (e.g.
`state.verify_attestations`) so the policy can't wire to the wrong one;
sha256 over utf-8; branch-rename edge (episode_install_branch renames on
collision): match attestations primarily by content_hash, with branch
recorded for observability.

**Confirmed by review (no change needed):** the zero-tool-call submit
path already exists and is tested (`episodes.py:860-865`,
`test_episodes.py:685`) — state the Part-1 conditional affirmatively;
`meta_declare_unsolved` bypass is the base-class default; the
fallback_declared bypass is structural (`loop_v3.py:610` constructs the
declaration directly, never calls the tool); v2 stays byte-identical.
