# V3 Candidate Selection Report

Ground truth is used only for post-hoc grading after candidate generation and selection.

## Cross-Run Findings

- 1 run(s) made historical decisions on a boundary-flattened candidate. Their old verifier verdicts do not evaluate the corrected rendering.
- Best generated post-hoc character accuracy ranges from 91.0% to 96.5% (5.5% spread), so candidate-generation variance remains material after the rendering fix.

## borg_single_B_borg_0109v / 8d5bce9769b1

- Status: `unsolved`
- Selected branch: `nullmask_main`
- Post-hoc best branch(es): `nullmask_main`, `wide_nullmask_result`
- Canonical source words: 78

- **Finding:** Historical snapshots flattened canonical source boundaries for null-mask branch(es): nullmask_main.
- **Finding:** Historical verifier rejection(s) were bound to the unsegmented content for: nullmask_main. Corrected renderings require fresh verification.
- **Finding:** Post-hoc best nullmask_main, wide_nullmask_result ranked 3 by the ground-truth-free scalar ordering.

| Scalar | GT | Branch | Roles | Renderer | Boundaries | Dict | Quad | GT char | GT word | Verify |
|---:|---:|---|---|---|---|---:|---:|---:|---:|---|
| 1 | 3 | `anneal_simple_1` | - | `decoded_text_v1` | source | 0.474 | -5.234 | 91.3% | 67.9% | reject |
| 2 | 3 | `wide_simple_result` | - | `decoded_text_v1` | source | 0.474 | -5.257 | 91.3% | 67.9% | - |
| 3 | 1 | `nullmask_main` | best_scored_branch, workflow_branch, declared_or_selected_branch | `key_with_null_mask_v1` | source | 0.474 | -5.303 | 96.5% | 82.1% | - |
| 3 | 1 | `wide_nullmask_result` | - | `key_with_null_mask_v1` | source | 0.474 | -5.303 | 96.5% | 82.1% | - |
| 5 | 5 | `automated_preflight` | - | `decoded_text_v1` | source | 0.372 | -5.339 | 84.9% | 52.6% | reject |
| 6 | 6 | `homophonic_anneal_1` | - | `decoded_text_v1` | source | 0.218 | -5.845 | 40.0% | 0.0% | - |

### Boundary Replay

| Branch | Unsegmented GT word | Boundary-preserved GT word | GT char | Historical verify | Historical loss |
|---|---:|---:|---:|---|---|
| `nullmask_main` | 0.0% | 82.1% | 96.5% | reject | yes |
| `wide_nullmask_result` | 0.0% | 82.1% | 96.5% | reject | no |

### Best-Branch Timeline

| Turn | Branch | Dict | Quad | Boundaries |
|---:|---|---:|---:|---|
| 1 | `automated_preflight` | 0.372 | -5.339 | yes |
| 3 | `anneal_simple_1` | 0.474 | -5.234 | yes |
| 14 | `nullmask_main` | 0.684 | -5.608 | no |

## borg_single_B_borg_0109v / ac129831aebc

- Status: `unsolved`
- Selected branch: `hillclimb_subst_1`
- Post-hoc best branch(es): `automated_preflight`, `hillclimb_subst_1`
- Canonical source words: 78

| Scalar | GT | Branch | Roles | Renderer | Boundaries | Dict | Quad | GT char | GT word | Verify |
|---:|---:|---|---|---|---|---:|---:|---:|---:|---|
| 1 | 1 | `hillclimb_subst_1` | best_scored_branch, workflow_branch, declared_or_selected_branch | `decoded_text_v1` | source | 0.500 | -5.262 | 91.0% | 66.7% | reject |
| 2 | 1 | `automated_preflight` | - | `decoded_text_v1` | source | 0.474 | -5.271 | 91.0% | 66.7% | reject |
| 3 | 4 | `auto_subst_transform_1` | - | `decoded_text_v1` | source | 0.436 | -5.290 | 89.6% | 61.5% | - |
| 4 | 4 | `wide_targeted_repair_1` | - | `decoded_text_v1` | source | 0.436 | -5.307 | 89.6% | 61.5% | - |
| 5 | 3 | `legacy_wide_alt_1` | - | `decoded_text_v1` | source | 0.436 | -5.314 | 89.6% | 62.8% | - |
| 6 | 4 | `broad_subst_1` | - | `decoded_text_v1` | source | 0.436 | -5.317 | 89.6% | 61.5% | - |
| 7 | 7 | `legacy_wide_alt_2` | - | `decoded_text_v1` | source | 0.397 | -5.305 | 77.7% | 37.2% | - |

### Best-Branch Timeline

| Turn | Branch | Dict | Quad | Boundaries |
|---:|---|---:|---:|---|
| 1 | `automated_preflight` | 0.474 | -5.271 | yes |
| 3 | `hillclimb_subst_1` | 0.500 | -5.262 | yes |
