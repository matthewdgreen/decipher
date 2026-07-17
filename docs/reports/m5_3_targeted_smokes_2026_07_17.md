# M5.3 Targeted Smoke Results - 2026-07-17

These paid smokes used OpenAI `gpt-5.5`. Ground truth was available only to the
outer benchmark scorer after each run; it was not available to agent routing,
verification, repair, branch selection, or declaration.

## Spend

| Run | Cost | Configured cutoff | Outcome |
|---|---:|---:|---|
| Borg repair path, initial | $2.7135 | $5.00 | 96.5% char, honest unsolved |
| Borg repair path, binding fix | $2.2217 | $5.00 | 91.0% char / 66.7% word, honest unsolved |
| No-boundary exact negative control | $0.6321 | $2.50 | 100% char, correctly not declared |
| Word-boundary positive control | $0.0846 | $2.50 | 100% char / 100% word, declared turn 2 |
| Cost-cutoff trip | $0.1326 | $0.10 | stopped before turn 3; $0.0326 final-call overshoot |
| **Total** | **$5.7845** |  |  |

## Repair Path

The initial Borg run produced a usable Latin reading at turn 4 and launched a
repair transaction at turn 5. The worker created a host-supported improved
fork, but returned a prose sentence in the result's `edits` array instead of
copying the exact host label `W:D->F`. The host correctly failed closed with
`unsupported_edit_claim`.

The resulting fix:

- tells the worker to return exact host labels only;
- accepts legacy prose only when every embedded mapping-shaped label is exact
  host evidence; and
- still rejects label-free prose or any claim containing an invented mapping.

The fixed rerun bound its claim successfully and passed winner, fork-evidence,
adjudication, collateral, and no-op checks. The host then rejected the change
for a genuine scalar regression (`dict_rate -0.0128`, quadgram `-0.0054`). This
is the intended default-deny behavior and is a reasoned no-safe-repair result,
not a process failure.

Both runs verified by turn 1, produced a reading by turn 4, attempted repair by
turn 5, respected episode execution caps, used valid experiment configs, and
terminated honestly. The first run met the character target but selected a
no-boundary finalist with 0% post-hoc word score. The fixed rerun missed the
pre-registered accuracy targets (91.0% < 93%; 66.7% < 70%). Per the master
specification, this is now a search/selection-quality question rather than a
reason to enlarge worker budgets.

## Diplomatic Verification

The first exact control used a no-boundary synthetic. Its plaintext was exactly
correct by post-hoc character score, but the verifier rejected it as a complete
solution because the rendered candidate lacked word boundaries and contained
telegraphic article omissions. Two declaration attempts were blocked. This is
a useful negative control, but it was an unsuitable positive control.

A deterministic word-boundary substitution control then produced an exact
preflight plaintext. Verification was positive on turn 1 and the lead declared
on turn 2 at 100% character and word accuracy. This establishes both sides of
the diplomatic gate without weakening it for unsegmented text.

## Cost Cutoff

With `max_cost_usd=0.10`, the run completed two lead sends and one verification
send, then emitted `cost_ceiling_reached` before turn 3. The queued search
episode executed zero worker calls and no later paid send occurred. Final cost
was $0.1326 because the last call began below the cutoff and its cost became
known only after completion. Artifacts and the analyzer now describe this as a
pre-send cutoff and display final-call overshoot explicitly.

## Confidence And Next Gate

M5.3's local mechanics, reading production, repair default-deny checks,
diplomatic positive/negative gating, artifact parity, and paid cutoff behavior
are now directly exercised. The focused six-run Stage-1 packet should still
wait: the fixed Borg smoke missed both accuracy thresholds, and the two Borg
runs showed meaningful basin/selection variance. The next work should diagnose
why the strongest solver basin and word-boundary-preserving basin are not
selected reliably; it should not increase reading budgets or launch the full
M6 bake-off.

## Artifacts

- `artifacts/m5_3_targeted_smokes/repair_path/.../8d5bce9769b1.json`
- `artifacts/m5_3_targeted_smokes/repair_path_fixed/.../ac129831aebc.json`
- `artifacts/m5_3_targeted_smokes/positive_declaration/.../d65f5f4876a7.json`
- `artifacts/m5_3_targeted_smokes/positive_word_boundary/.../29b8f89ad6ee.json`
- `artifacts/m5_3_targeted_smokes/cost_ceiling/.../b0f1b0927e21.json`
